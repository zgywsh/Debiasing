import argparse
import sys
import time
import face_recognition
from datetime import datetime
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
import ot
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from torchvision import transforms
import itertools
import json
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from packaging import version
from diffusers.utils.import_utils import is_xformers_available
import os, sys
from pathlib import Path

import argparse
import itertools
import logging
import math
import shutil
import json
import pytz
import random
from datetime import datetime
from tqdm.auto import tqdm
import copy
import pickle as pkl
import yaml
from packaging import version
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
from torch import nn
import torchvision
from torchvision.models.mobilenetv3 import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import scipy
from skimage import transform
import kornia
from sentence_transformers import SentenceTransformer, util

import transformers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.training_utils import EMAModel


# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis
import warnings


warnings.filterwarnings("ignore")


class FaceFeatsModel(torch.nn.Module):
    def __init__(self, face_feats_path):
        super().__init__()

        with open(face_feats_path, "rb") as f:
            face_feats, face_genders, face_logits = pkl.load(f)

        face_feats = torch.nn.functional.normalize(face_feats, dim=-1)
        self.face_feats = nn.Parameter(face_feats)
        self.face_feats.requires_grad_(False)

    def forward(self, x):
        """no forward function"""
        return None

    @torch.no_grad()
    def semantic_search(self, query_embeddings, selector=None, return_similarity=False):
        """search the closest face embedding from vector database."""
        target_embeddings = torch.ones_like(query_embeddings) * (-1)
        if return_similarity:
            similarities = torch.ones(
                [query_embeddings.shape[0]],
                device=query_embeddings.device,
                dtype=query_embeddings.dtype,
            ) * (-1)

        if selector.sum() > 0:
            hits = util.semantic_search(
                query_embeddings[selector],
                self.face_feats,
                score_function=util.dot_score,
                top_k=1,
            )
            target_embeddings_ = torch.cat(
                [self.face_feats[hit[0]["corpus_id"]].unsqueeze(dim=0) for hit in hits]
            )
            target_embeddings[selector] = target_embeddings_
            if return_similarity:
                similarities_ = torch.tensor(
                    [hit[0]["score"] for hit in hits],
                    device=query_embeddings.device,
                    dtype=query_embeddings.dtype,
                )
                similarities[selector] = similarities_

        if return_similarity:
            return target_embeddings.data.detach().clone(), similarities
        else:
            return target_embeddings.data.detach().clone()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def plot_in_grid(
    images,
    save_to,
    face_indicators=None,
    face_bboxs=None,
    preds_gender=None,
    pred_class_probs_gender=None,
):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    images_w_face = images[face_indicators]
    images_wo_face = images[face_indicators.logical_not()]

    # first reorder everything from most to least male, from most to least female, and finally images without faces
    idxs_male = (preds_gender == 1).nonzero(as_tuple=False).view([-1])
    probs_male = pred_class_probs_gender[idxs_male]
    idxs_male = idxs_male[probs_male.argsort(descending=True)]

    idxs_female = (preds_gender == 0).nonzero(as_tuple=False).view([-1])
    probs_female = pred_class_probs_gender[idxs_female]
    idxs_female = idxs_female[probs_female.argsort(descending=True)]

    idxs_no_face = (preds_gender == -1).nonzero(as_tuple=False).view([-1])

    images_to_plot = []
    idxs_reordered = torch.torch.cat([idxs_male, idxs_female, idxs_no_face])

    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]

        if pred_gender == 1:
            pred = "Male"
            border_color = "blue"
        elif pred_gender == 0:
            pred = "Female"
            border_color = "red"
        elif pred_gender == -1:
            pred = "Undetected"
            border_color = "white"

        img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)
        img_pil_draw.rectangle(
            face_bbox.tolist(), fill=None, outline=border_color, width=4
        )

        img_pil = ImageOps.expand(img_pil, border=(50, 0, 0, 0), fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle(
                [(0, 0), (50, (1 - pred_class_prob_gender.item()) * 512)],
                fill="white",
                outline=None,
            )

        fnt = ImageFont.truetype(
            font="/finetune/data/0-utils/arial-bold.ttf",
            size=100,
        )
        img_pil_draw.text((400, 400), f"{idx.item()}", align="left", font=fnt)

        img_pil = ImageOps.expand(
            img_pil_draw._image, border=(10, 10, 10, 10), fill="black"
        )

        images_to_plot.append(img_pil)

    N_imgs = len(images_to_plot)
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    for i in range(N1 * N2 - N_imgs):
        images_to_plot.append(
            Image.new("RGB", color="white", size=images_to_plot[0].size)
        )
    grid = image_grid(images_to_plot, N1, N2)
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    grid.save(save_to, quality=25)


def make_grad_hook(coef):
    return lambda x: coef * x


def customized_all_gather(tensor, return_tensor_other_processes=False):
    tensor_all = [tensor.detach().clone() for _ in range(1)]

    tensor_all = torch.cat(tensor_all, dim=0)

    if return_tensor_other_processes:
        tensor_others = torch.empty(
            [0] + list(tensor_all.shape[1:]), device=tensor.device, dtype=tensor.dtype
        )
        return tensor_all, tensor_others
    else:
        return tensor_all


def expand_bbox(bbox, expand_coef, target_ratio):
    """
    bbox: [width_small, height_small, width_large, height_large],
        this is the format returned from insightface.app.FaceAnalysis
    expand_coef: 0 is no expansion
    target_ratio: target img height/width ratio

    note that it is possible that bbox is outside the original image size
    confirmed for insightface.app.FaceAnalysis
    """

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    current_ratio = bbox_height / bbox_width
    if current_ratio > target_ratio:
        more_height = bbox_height * expand_coef
        more_width = (bbox_height + more_height) / target_ratio - bbox_width
    elif current_ratio <= target_ratio:
        more_width = bbox_width * expand_coef
        more_height = (bbox_width + more_width) * target_ratio - bbox_height

    bbox_new = [0, 0, 0, 0]
    bbox_new[0] = int(round(bbox[0] - more_width * 0.5))
    bbox_new[2] = int(round(bbox[2] + more_width * 0.5))
    bbox_new[1] = int(round(bbox[1] - more_height * 0.5))
    bbox_new[3] = int(round(bbox[3] + more_height * 0.5))
    return bbox_new


def crop_face(img_tensor, bbox_new, target_size, fill_value):
    """
    img_tensor: [3,H,W]
    bbox_new: [width_small, height_small, width_large, height_large]
    target_size: [width,height]
    fill_value: value used if need to pad
    """
    img_height, img_width = img_tensor.shape[-2:]

    idx_left = max(bbox_new[0], 0)
    idx_right = min(bbox_new[2], img_width)
    idx_bottom = max(bbox_new[1], 0)
    idx_top = min(bbox_new[3], img_height)

    pad_left = max(-bbox_new[0], 0)
    pad_right = max(-(img_width - bbox_new[2]), 0)
    pad_top = max(-bbox_new[1], 0)
    pad_bottom = max(-(img_height - bbox_new[3]), 0)

    img_face = img_tensor[:, idx_bottom:idx_top, idx_left:idx_right]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_face = torchvision.transforms.Pad(
            [pad_left, pad_top, pad_right, pad_bottom], fill=fill_value
        )(img_face)
    img_face = torchvision.transforms.Resize(size=target_size, antialias=True)(img_face)
    return img_face


def image_pipeline(img, tgz_landmark):
    img = (img + 1) / 2.0 * 255  # map to [0,255]

    crop_size = (112, 112)
    src_landmark = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left corner of the mouth
            [70.7299, 92.2041],
        ]  # right corner of the mouth
    )

    tform = transform.SimilarityTransform()
    tform.estimate(tgz_landmark, src_landmark)

    M = torch.tensor(tform.params[0:2, :]).unsqueeze(dim=0).to(img.dtype).to(img.device)
    img_face = kornia.geometry.transform.warp_affine(
        img.unsqueeze(dim=0),
        M,
        crop_size,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_face = img_face.squeeze()

    img_face = (img_face / 255.0) * 2 - 1  # map back to [-1,1]
    return img_face


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Script to finetune Stable Diffusion for debiasing purposes."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/runwayml/sd-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_text_encoder_lora_from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_unet_lora_from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_prefix_embedding_from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--number_prefix_tokens",
        type=int,
        default=5,
        help="number of tokens as prefix, must be provided when --load_prefix_embedding_from is provided",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="/outputs/occupation.json",
        # required=True,
    )
    parser.add_argument(
        "--num_imgs_per_prompt",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/train_outputs",
        # required=True
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1997,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="provide the checkpoint path to resume from checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=50,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--guidance_scale",
        help="diffusion model text guidance scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--num_denoising_steps",
        help="num denoising steps used for image generation",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch_size", help="batch size for image generation", type=int, default=10
    )
    parser.add_argument(
        "--gender_classifier_weight_path",
        default="/finetune/data/5-trained-test-classifiers/CelebA-MobileNetLarge-Gender-09191318/epoch=19-step=25320_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--race_classifier_weight_path",
        default="/finetune/data/5-trained-test-classifiers/fairface-MobileNetLarge-Race4-09191318/epoch=19-step=6760_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--opensphere_config",
        help="train, val, test batch size",
        type=str,
        default="/finetune/data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/config.yml",
    )
    parser.add_argument(
        "--opensphere_model_path",
        help="train, val, test batch size",
        type=str,
        default="/finetune/data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/models/backbone_100000.pth",
    )
    parser.add_argument(
        "--face_feats_path",
        help="external face feats, used for the face realism preserving loss",
        type=str,
        default="/finetune/data/3-face-features/CelebA_MobileNetLarge_08240859/face_feats.pkl",
    )
    parser.add_argument(
        "--size_face",
        type=int,
        default=224,
        help="faces will be resized to this size",
    )
    parser.add_argument(
        "--size_aligned_face",
        type=int,
        default=112,
        help="aligned faces will be resized to this size",
    )
    parser.add_argument(
        "--train_plot_every_n_iter",
        help="plot training stats every n iteration",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--uncertainty_threshold",
        help="the uncertainty threshold used in distributional alignment loss",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--train_GPU_batch_size",
        help="training batch size in every GPU",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--face_gender_confidence_level",
        help="train, val, test batch size",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight_loss_face",
        default=1,
        help="weight for the face realism preserving loss",
        type=float,
    )
    parser.add_argument(
        "--img_size_small",
        type=int,
        default=224,
        help="For some operations, images will be resized to this size for more efficient processing",
    )
    parser.add_argument(
        "--factor1", help="train, val, test batch size", type=float, default=0.2
    )
    parser.add_argument(
        "--factor2", help="train, val, test batch size", type=float, default=0.2
    )
    parser.add_argument(
        "--weight_loss_img",
        default=8,
        help="weight for the image semantics preserving loss",
        type=float,
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


args = parse_args()


@torch.no_grad()
def generate_dynamic_targets(probs, target_ratio=0.5, w_uncertainty=False):
    """generate dynamic targets for the distributional alignment loss

    Args:
        probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
        target_ratio (float): target distribution, the percentage of class 1 (male)
        w_uncertainty (True/False): whether return uncertainty measures

    Returns:
        targets_all (torch.tensor): target classes
        uncertainty_all (torch.tensor): uncertainty of target classes
    """
    idxs_2_rank = (probs != -1).all(dim=-1)
    probs_2_rank = probs[idxs_2_rank]

    rank = torch.argsort(torch.argsort(probs_2_rank[:, 1]))
    targets = (rank >= (rank.shape[0] * target_ratio)).long()

    targets_all = torch.ones(
        [probs.shape[0]], dtype=torch.long, device=probs.device
    ) * (-1)
    targets_all[idxs_2_rank] = targets

    if w_uncertainty:
        uncertainty = torch.ones(
            [probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device
        ) * (-1)
        uncertainty[targets == 1] = (
            torch.tensor(
                1
                - scipy.stats.binom.cdf(
                    (rank[targets == 1]).cpu().numpy(),
                    probs_2_rank.shape[0],
                    1 - target_ratio,
                )
            )
            .to(probs.dtype)
            .to(probs.device)
        )
        uncertainty[targets == 0] = (
            torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets == 0].cpu().numpy(),
                    probs_2_rank.shape[0],
                    target_ratio,
                )
            )
            .to(probs.dtype)
            .to(probs.device)
        )

        uncertainty_all = torch.ones(
            [probs.shape[0]], dtype=probs.dtype, device=probs.device
        ) * (-1)
        uncertainty_all[idxs_2_rank] = uncertainty

        return targets_all, uncertainty_all
    else:
        return targets_all


face_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    # providers=["CUDAExecutionProvider"],
    # provider_options=[{"device_id": device.index}],
)
face_app.prepare(ctx_id=1, det_size=(640, 640))

gender_classifier = mobilenet_v3_large(
    weights=MobileNet_V3_Large_Weights.DEFAULT,
    width_mult=1.0,
    reduced_tail=False,
    dilated=False,
)
gender_classifier._modules["classifier"][3] = nn.Linear(1280, 2, bias=True)
gender_classifier.load_state_dict(torch.load(args.gender_classifier_weight_path))

race_classifier = mobilenet_v3_large(
    weights=MobileNet_V3_Large_Weights.DEFAULT,
    width_mult=1.0,
    reduced_tail=False,
    dilated=False,
)
race_classifier._modules["classifier"][3] = nn.Linear(1280, 4, bias=True)
race_classifier.load_state_dict(torch.load(args.race_classifier_weight_path))
face_feats_model = FaceFeatsModel(args.face_feats_path)

# build opensphere model
sys.path.append(Path(__file__).parent.parent.__str__())
sys.path.append(Path(__file__).parent.parent.joinpath("opensphere").__str__())
from opensphere.builder import build_from_cfg
from opensphere.utils import fill_config

with open(args.opensphere_config, "r") as f:
    opensphere_config = yaml.load(f, yaml.SafeLoader)
opensphere_config["data"] = fill_config(opensphere_config["data"])
face_feats_net = build_from_cfg(
    opensphere_config["model"]["backbone"]["net"],
    "model.backbone",
)
weight_dtype_high_precision = torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
face_feats_net = nn.DataParallel(face_feats_net)
face_feats_net.load_state_dict(torch.load(args.opensphere_model_path))
face_feats_net = face_feats_net.module

device = "cuda:1"
gender_classifier.to(device, dtype=weight_dtype)
gender_classifier.requires_grad_(False)
gender_classifier.eval()
race_classifier.to(device, dtype=weight_dtype)
race_classifier.requires_grad_(False)
race_classifier.eval()
# set up face_recognition and face_app on all devices

CE_loss = nn.CrossEntropyLoss(reduction="none")
face_feats_net.to(device)
face_feats_net.requires_grad_(False)
face_feats_net.to(weight_dtype)
face_feats_net.eval()
face_feats_model.to(weight_dtype_high_precision)
face_feats_model.to(device)
face_feats_model.eval()


clip_image_processoor = CLIPImageProcessor.from_pretrained(
    "/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
)
clip_vision_model_w_proj = CLIPModel.from_pretrained("/clip-vit-base-patch32")
clip_vision_model_w_proj.vision_model.to(device, dtype=weight_dtype)
clip_vision_model_w_proj.visual_projection.to(device, dtype=weight_dtype)
clip_vision_model_w_proj.requires_grad_(False)
clip_vision_model_w_proj.gradient_checkpointing_enable()


def get_face_feats(net, data, flip=True, normalize=True, to_high_precision=True):
    # extract features from the original
    # and horizontally flipped data
    feats = net(data)
    if flip:
        data = torch.flip(data, [3])
        feats += net(data)
    if to_high_precision:
        feats = feats.to(torch.float)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


def get_face(images, fill_value=-1):
    """
    images:shape [N,3,H,W], in range [-1,1], pytorch tensor
    returns:
        face_indicators: torch tensor of shape [N], only True or False
            True means face is detected, False otherwise
        face_bboxs: torch tensor of shape [N,4],
            if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
        face_chips: torch tensor of shape [N,3,224,224]
            if face_indicator is False, the corresponding face_chip will be all fill_value
    """
    (
        face_indicators_app,
        face_bboxs_app,
        face_chips_app,
        face_landmarks_app,
        aligned_face_chips_app,
    ) = get_face_app(images, fill_value=fill_value)

    if face_indicators_app.logical_not().sum() > 0:
        (
            face_indicators_FR,
            face_bboxs_FR,
            face_chips_FR,
            face_landmarks_FR,
            aligned_face_chips_FR,
        ) = get_face_FR(
            images[face_indicators_app.logical_not()], fill_value=fill_value
        )

        face_bboxs_app[face_indicators_app.logical_not()] = face_bboxs_FR
        face_chips_app[face_indicators_app.logical_not()] = face_chips_FR
        face_landmarks_app[face_indicators_app.logical_not()] = face_landmarks_FR
        aligned_face_chips_app[face_indicators_app.logical_not()] = (
            aligned_face_chips_FR
        )

        face_indicators_app[face_indicators_app.logical_not()] = face_indicators_FR

    return (
        face_indicators_app,
        face_bboxs_app,
        face_chips_app,
        face_landmarks_app,
        aligned_face_chips_app,
    )


def get_largest_face_FR(faces_from_FR, dim_max, dim_min):
    if len(faces_from_FR) == 1:
        return faces_from_FR[0]
    elif len(faces_from_FR) > 1:
        area_max = 0
        idx_max = 0
        for idx, bbox in enumerate(faces_from_FR):
            bbox1 = np.array((bbox[-1],) + bbox[:-1])
            area = (min(bbox1[2], dim_max) - max(bbox1[0], dim_min)) * (
                min(bbox1[3], dim_max) - max(bbox1[1], dim_min)
            )
            if area > area_max:
                area_max = area
                idx_max = idx
        return faces_from_FR[idx_max]


def get_face_FR(images, fill_value=-1):
    """
    images:shape [N,3,H,W], in range [-1,1], pytorch tensor
    returns:
        face_indicators: torch tensor of shape [N], only True or False
            True means face is detected, False otherwise
        face_bboxs: torch tensor of shape [N,4],
            if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
        face_chips: torch tensor of shape [N,3,224,224]
            if face_indicator is False, the corresponding face_chip will be all fill_value
    """

    images_np = (
        ((images * 0.5 + 0.5) * 255)
        .cpu()
        .detach()
        .permute(0, 2, 3, 1)
        .float()
        .numpy()
        .astype(np.uint8)
    )

    face_indicators_FR = []
    face_bboxs_FR = []
    face_chips_FR = []
    face_landmarks_FR = []
    aligned_face_chips_FR = []
    for idx, image_np in enumerate(images_np):
        # import pdb; pdb.set_trace()
        faces_from_FR = face_recognition.face_locations(
            image_np
            # , number_of_times_to_upsample=0, model="cnn"
        )
        if len(faces_from_FR) == 0:
            face_indicators_FR.append(False)
            face_bboxs_FR.append([fill_value] * 4)
            face_chips_FR.append(
                torch.ones(
                    [1, 3, args.size_face, args.size_face],
                    dtype=images.dtype,
                    device=images.device,
                )
                * (fill_value)
            )
            face_landmarks_FR.append(
                torch.ones([1, 5, 2], dtype=images.dtype, device=images.device)
                * (fill_value)
            )
            aligned_face_chips_FR.append(
                torch.ones(
                    [1, 3, args.size_aligned_face, args.size_aligned_face],
                    dtype=images.dtype,
                    device=images.device,
                )
                * (fill_value)
            )
        else:
            face_from_FR = get_largest_face_FR(
                faces_from_FR, dim_max=image_np.shape[0], dim_min=0
            )
            bbox = face_from_FR
            bbox = np.array(
                (bbox[-1],) + bbox[:-1]
            )  # need to convert bbox from face_recognition to the right order
            bbox = expand_bbox(
                bbox, expand_coef=1.1, target_ratio=1
            )  # need to use a larger expand_coef for FR
            face_chip = crop_face(
                images[idx],
                bbox,
                target_size=[args.size_face, args.size_face],
                fill_value=fill_value,
            )

            face_landmarks = face_recognition.face_landmarks(
                image_np, face_locations=[face_from_FR], model="large"
            )

            left_eye = np.array(face_landmarks[0]["left_eye"]).mean(axis=0)
            right_eye = np.array(face_landmarks[0]["right_eye"]).mean(axis=0)
            nose_tip = np.array(face_landmarks[0]["nose_bridge"][-1])
            top_lip_left = np.array(face_landmarks[0]["top_lip"][0])
            top_lip_right = np.array(face_landmarks[0]["top_lip"][6])
            face_landmarks = np.stack(
                [left_eye, right_eye, nose_tip, top_lip_left, top_lip_right]
            )

            aligned_face_chip = image_pipeline(images[idx], face_landmarks)

            face_indicators_FR.append(True)
            face_bboxs_FR.append(bbox)
            face_chips_FR.append(face_chip.unsqueeze(dim=0))
            face_landmarks_FR.append(
                torch.tensor(face_landmarks)
                .unsqueeze(dim=0)
                .to(device=images.device)
                .to(images.dtype)
            )
            aligned_face_chips_FR.append(aligned_face_chip.unsqueeze(dim=0))

    face_indicators_FR = torch.tensor(face_indicators_FR).to(device=images.device)
    face_bboxs_FR = torch.tensor(face_bboxs_FR).to(device=images.device)
    face_chips_FR = torch.cat(face_chips_FR, dim=0)
    face_landmarks_FR = torch.cat(face_landmarks_FR, dim=0)
    aligned_face_chips_FR = torch.cat(aligned_face_chips_FR, dim=0)

    return (
        face_indicators_FR,
        face_bboxs_FR,
        face_chips_FR,
        face_landmarks_FR,
        aligned_face_chips_FR,
    )


def get_largest_face_app(face_from_app, dim_max, dim_min):
    if len(face_from_app) == 1:
        return face_from_app[0]
    elif len(face_from_app) > 1:
        area_max = 0
        idx_max = 0
        for idx in range(len(face_from_app)):
            bbox = face_from_app[idx]["bbox"]
            area = (min(bbox[2], dim_max) - max(bbox[0], dim_min)) * (
                min(bbox[3], dim_max) - max(bbox[1], dim_min)
            )
            if area > area_max:
                area_max = area
                idx_max = idx
        return face_from_app[idx_max]


def get_face_app(images, fill_value=-1):
    """
    images:shape [N,3,H,W], in range [-1,1], pytorch tensor
    returns:
        face_indicators: torch tensor of shape [N], only True or False
            True means face is detected, False otherwise
        face_bboxs: torch tensor of shape [N,4],
            if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
        face_chips: torch tensor of shape [N,3,224,224]
            if face_indicator is False, the corresponding face_chip will be all fill_value
    """
    images_np = (
        ((images * 0.5 + 0.5) * 255)
        .cpu()
        .detach()
        .permute(0, 2, 3, 1)
        .float()
        .numpy()
        .astype(np.uint8)
    )

    face_indicators_app = []
    face_bboxs_app = []
    face_chips_app = []
    face_landmarks_app = []
    aligned_face_chips_app = []
    for idx, image_np in enumerate(images_np):
        # face_app.get input should be [BGR]
        faces_from_app = face_app.get(image_np[:, :, [2, 1, 0]])
        if len(faces_from_app) == 0:
            face_indicators_app.append(False)
            face_bboxs_app.append([fill_value] * 4)
            face_chips_app.append(
                torch.ones(
                    [1, 3, args.size_face, args.size_face],
                    dtype=images.dtype,
                    device=images.device,
                )
                * (fill_value)
            )
            face_landmarks_app.append(
                torch.ones([1, 5, 2], dtype=images.dtype, device=images.device)
                * (fill_value)
            )
            aligned_face_chips_app.append(
                torch.ones(
                    [1, 3, args.size_aligned_face, args.size_aligned_face],
                    dtype=images.dtype,
                    device=images.device,
                )
                * (fill_value)
            )
        else:
            face_from_app = get_largest_face_app(
                faces_from_app, dim_max=image_np.shape[0], dim_min=0
            )
            bbox = expand_bbox(face_from_app["bbox"], expand_coef=0.5, target_ratio=1)
            face_chip = crop_face(
                images[idx],
                bbox,
                target_size=[args.size_face, args.size_face],
                fill_value=fill_value,
            )

            face_landmarks = np.array(face_from_app["kps"])
            aligned_face_chip = image_pipeline(images[idx], face_landmarks)

            face_indicators_app.append(True)
            face_bboxs_app.append(bbox)
            face_chips_app.append(face_chip.unsqueeze(dim=0))
            face_landmarks_app.append(
                torch.tensor(face_landmarks)
                .unsqueeze(dim=0)
                .to(device=images.device)
                .to(images.dtype)
            )
            aligned_face_chips_app.append(aligned_face_chip.unsqueeze(dim=0))

    face_indicators_app = torch.tensor(face_indicators_app).to(device=images.device)
    face_bboxs_app = torch.tensor(face_bboxs_app).to(device=images.device)
    face_chips_app = torch.cat(face_chips_app, dim=0)
    face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
    aligned_face_chips_app = torch.cat(aligned_face_chips_app, dim=0)

    return (
        face_indicators_app,
        face_bboxs_app,
        face_chips_app,
        face_landmarks_app,
        aligned_face_chips_app,
    )


def get_face_gender(face_chips, selector=None, fill_value=-1):
    """for CelebA classifier"""
    if selector != None:
        face_chips_w_faces = face_chips[selector]
    else:
        face_chips_w_faces = face_chips

    if face_chips_w_faces.shape[0] == 0:
        logits_gender = torch.empty(
            [0, 2], dtype=face_chips.dtype, device=face_chips.device
        )
        probs_gender = torch.empty(
            [0, 2], dtype=face_chips.dtype, device=face_chips.device
        )
        # pred_class_probs_gender = torch.empty([0], dtype=face_chips.dtype, device=face_chips.device)
        preds_gender = torch.empty([0], dtype=torch.int64, device=face_chips.device)
    else:
        # logits = gender_classifier(face_chips_w_faces)
        # logits_gender = logits.view([logits.shape[0], -1, 2])[:, 20, :]
        logits_gender = gender_classifier(face_chips_w_faces)
        probs_gender = torch.softmax(logits_gender, dim=-1)

        temp = probs_gender.max(dim=-1)
        # pred_class_probs_gender = temp.values
        preds_gender = temp.indices

    if selector != None:
        preds_gender_new = torch.ones(
            [selector.shape[0]] + list(preds_gender.shape[1:]),
            dtype=preds_gender.dtype,
            device=preds_gender.device,
        ) * (fill_value)
        preds_gender_new[selector] = preds_gender

        probs_gender_new = torch.ones(
            [selector.shape[0]] + list(probs_gender.shape[1:]),
            dtype=probs_gender.dtype,
            device=probs_gender.device,
        ) * (fill_value)
        probs_gender_new[selector] = probs_gender

        logits_gender_new = torch.ones(
            [selector.shape[0]] + list(logits_gender.shape[1:]),
            dtype=logits_gender.dtype,
            device=logits_gender.device,
        ) * (fill_value)
        logits_gender_new[selector] = logits_gender

        return preds_gender_new, probs_gender_new, logits_gender_new
    else:
        return preds_gender, probs_gender, logits_gender


def get_face_race(face_chips, selector=None, fill_value=-1):
    """for CelebA classifier"""
    if selector != None:
        face_chips_w_faces = face_chips[selector]
    else:
        face_chips_w_faces = face_chips

    if face_chips_w_faces.shape[0] == 0:
        logits_race = torch.empty(
            [0, 4], dtype=face_chips.dtype, device=face_chips.device
        )
        probs_race = torch.empty(
            [0, 4], dtype=face_chips.dtype, device=face_chips.device
        )
        # pred_class_probs_gender = torch.empty([0], dtype=face_chips.dtype, device=face_chips.device)
        preds_race = torch.empty([0], dtype=torch.int64, device=face_chips.device)
    else:
        logits = race_classifier(face_chips_w_faces)
        logits_race = logits
        probs_race = torch.softmax(logits_race, dim=-1)

        temp = probs_race.max(dim=-1)
        # pred_class_probs_gender = temp.values
        preds_race = temp.indices

    if selector != None:
        preds_race_new = torch.ones(
            [selector.shape[0]] + list(preds_race.shape[1:]),
            dtype=preds_race.dtype,
            device=preds_race.device,
        ) * (fill_value)
        preds_race_new[selector] = preds_race

        probs_race_new = torch.ones(
            [selector.shape[0]] + list(probs_race.shape[1:]),
            dtype=probs_race.dtype,
            device=probs_race.device,
        ) * (fill_value)
        probs_race_new[selector] = probs_race

        logits_race_new = torch.ones(
            [selector.shape[0]] + list(logits_race.shape[1:]),
            dtype=logits_race.dtype,
            device=logits_race.device,
        ) * (fill_value)
        logits_race_new[selector] = logits_race

        return preds_race_new, probs_race_new, logits_race_new
    else:
        return preds_race, probs_race, logits_race


def gen_dynamic_weights(
    face_indicators, targets, preds_gender_ori, probs_gender_ori, factor=0.2
):
    weights = []
    for (
        face_indicator,
        target,
        pred_gender_ori,
        prob_gender_ori,
    ) in itertools.zip_longest(
        face_indicators, targets, preds_gender_ori, probs_gender_ori
    ):
        if (face_indicator == False).all():
            weights.append(1)
        else:
            if target == -1:
                weights.append(factor)
            elif target == pred_gender_ori:
                weights.append(1)
            elif target != pred_gender_ori:
                weights.append(factor)

    weights = torch.tensor(
        weights, dtype=probs_gender_ori.dtype, device=probs_gender_ori.device
    )
    return weights


@torch.no_grad()
def generate_dynamic_targets_race(probs, w_uncertainty=False):
    """generate dynamic targets for the distributional alignment loss

    Args:
        probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
        target_ratio (float): target distribution, the percentage of class 1 (male)
        w_uncertainty (True/False): whether return uncertainty measures

    Returns:
        targets_all (torch.tensor): target classes
        uncertainty_all (torch.tensor): uncertainty of target classes
    """
    idxs_2_rank = (probs != -1).all(dim=-1)
    probs_2_rank = probs[idxs_2_rank]

    a = np.ones([idxs_2_rank.sum()])  # source, uniform distribution on samples
    target_points = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    all_combs = []
    all_probs = []
    N = probs_2_rank.shape[0]
    for n1 in range(N + 1):
        for n2 in range(N - n1 + 1):
            for n3 in range(N - n1 - n2 + 1):
                n4 = N - n1 - n2 - n3
                all_combs.append([n1, n2, n3, n4])
                all_probs.append(
                    math.comb(N, n1)
                    * math.comb(N - n1, n2)
                    * math.comb(N - n1 - n2, n3)
                )

    all_combs = np.array(all_combs)
    all_probs = np.array(all_probs)
    all_probs = all_probs / np.linalg.norm(all_probs, ord=1)

    idxs_sorted = np.flip(all_probs.argsort())
    prob_accumulate = 0
    for i_idx, idx in enumerate(idxs_sorted):
        prob_accumulate += all_probs[idx]
        if prob_accumulate > 0.95:
            break
    all_combs = all_combs[idxs_sorted[: i_idx + 1]]
    all_probs = all_probs[idxs_sorted[: i_idx + 1]]

    M = ot.dist(
        np.array(probs_2_rank.cpu()), target_points, metric="euclidean", p=1
    )  # cost matrix
    target_probs = np.zeros([N, 4])
    for b, prob in itertools.zip_longest(all_combs, all_probs):
        T = ot.emd(a, b, M)
        target_probs += T * prob
    target_probs = target_probs / np.expand_dims(
        np.linalg.norm(target_probs, axis=-1, ord=1), axis=-1
    )

    targets = torch.tensor(target_probs.argmax(axis=-1))
    targets = targets.to(torch.long).to(probs.device)
    uncertainty = torch.tensor(1 - target_probs.max(axis=-1))
    uncertainty = uncertainty.to(probs.dtype).to(probs.device)

    targets_all = torch.ones(
        [probs.shape[0]], dtype=torch.long, device=probs.device
    ) * (-1)
    targets_all[idxs_2_rank] = targets

    if w_uncertainty:
        uncertainty_all = torch.ones(
            [probs.shape[0]], dtype=probs.dtype, device=probs.device
        ) * (-1)
        uncertainty_all[idxs_2_rank] = uncertainty

        return targets_all, uncertainty_all
    else:
        return targets_all


# args = parse_args()
def eval_images(images=None, images_ori=None, devide=False, test=False, iftargets=None):
    # torch.cuda.empty_cache()
    images = images.to(device)

    images_ori = images_ori.to(device)

    ######################################################gen_images
    (
        face_indicators_ori,
        face_bboxs_ori,
        face_chips_ori,
        face_landmarks_ori,
        aligned_face_chips_ori,
    ) = get_face(images_ori)

    face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = (
        get_face(images)
    )

    preds_gender, probs_gender, logits_gender = get_face_gender(
        face_chips, selector=face_indicators, fill_value=-1
    )
    preds_race, probs_race, logits_race = get_face_race(
        face_chips, selector=face_indicators, fill_value=-1
    )
    face_feats = torch.ones(
        [aligned_face_chips.shape[0], 512],
        dtype=weight_dtype_high_precision,
        device=aligned_face_chips.device,
    ) * (-1)
    if sum(face_indicators) > 0:
        face_feats_ = get_face_feats(
            face_feats_net, aligned_face_chips[face_indicators]
        )
        face_feats[face_indicators] = face_feats_

    probs_gender_all = probs_gender
    probs_race_all = probs_race

    # minus_one_count = (preds_gender == -1).sum().item()

    #################################################
    # Step 2: generate dynamic targets
    # also broadcast from process idx 0, just in case targets_all computed might be different on different processes
    targets_all1, uncertainty_all1 = generate_dynamic_targets(
        probs_gender_all, w_uncertainty=True
    )
    targets_all2, uncertainty_all2 = generate_dynamic_targets_race(
        probs_race_all, w_uncertainty=True
    )
    # targets_all[uncertainty_all > args.uncertainty_threshold] = -1
    targets1 = targets_all1
    targets2 = targets_all2
    if test == True:
        print(preds_gender, preds_race)
        # print(targets)
        return targets1, targets2
    if iftargets != None:
        (targets1, targets2) = iftargets

    #################################################
    # Step 3: generate all original images using the original diffusion model

    clip_img_mean = (
        torch.tensor(clip_image_processoor.image_mean)
        .reshape([-1, 1, 1])
        .to(device, dtype=weight_dtype)
    )  # mean is based on range [0,1]
    clip_img_std = (
        torch.tensor(clip_image_processoor.image_std)
        .reshape([-1, 1, 1])
        .to(device, dtype=weight_dtype)
    )

    def get_clip_feat(images, normalize=True, to_high_precision=True):
        """get clip features

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = ((images + 1) * 0.5 - clip_img_mean) / clip_img_std
        # embeds = clip_vision_model_w_proj(images_preprocessed).image_embeds
        embeds = clip_vision_model_w_proj.get_image_features(images_preprocessed)
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    preds_gender_ori, probs_gender_ori, logits_gender_ori = get_face_gender(
        face_chips_ori, selector=face_indicators_ori, fill_value=-1
    )
    preds_race_ori, probs_race_ori, logits_race_ori = get_face_race(
        face_chips_ori, selector=face_indicators_ori, fill_value=-1
    )
    images_small_ori = transforms.Resize(args.img_size_small)(images_ori)
    clip_feats_ori = get_clip_feat(
        images_small_ori, normalize=True, to_high_precision=True
    )
    face_feats_ori = get_face_feats(face_feats_net, aligned_face_chips_ori)
    loss_fair = []
    loss_face = []
    loss_clip = []

    targets = targets1
    # ################################################
    #     # Step 4: compute loss

    idxs_i = list(range(targets.shape[0]))
    N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
    for j in range(N_backward):
        idxs_ij = idxs_i[
            j * args.train_GPU_batch_size : (j + 1) * args.train_GPU_batch_size
        ]

        targets_ij = targets[idxs_ij]

        preds_gender_ori_ij = preds_gender_ori[idxs_ij]
        probs_gender_ori_ij = probs_gender_ori[idxs_ij]
        face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]
        face_feats_ori_ij = face_feats_ori[idxs_ij]
        clip_feats_ori_ij = clip_feats_ori[idxs_ij]

        images_ij = images[idxs_ij]
        (
            face_indicators_ij,
            face_bboxs_ij,
            face_chips_ij,
            face_landmarks_ij,
            aligned_face_chips_ij,
        ) = get_face(images_ij)
        preds_gender_ij, probs_gender_ij, logits_gender_ij = get_face_gender(
            face_chips_ij, selector=face_indicators_ij, fill_value=-1
        )

        idxs_w_face_loss = (
            ((face_indicators_ij == True) * (targets_ij != -1)).nonzero().view([-1])
        )
        loss_fair_ij_w_face_loss = CE_loss(
            logits_gender_ij[idxs_w_face_loss], targets_ij[idxs_w_face_loss]
        )
        loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=device) * (0)
        loss_fair_ij[idxs_w_face_loss] = loss_fair_ij_w_face_loss

        loss_face_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=device) * (0)
        idxs_w_face_feats_from_ori = (
            (
                (face_indicators_ij == True)
                * (targets_ij != -1)
                * (targets_ij == preds_gender_ori_ij)
                * (
                    probs_gender_ori_ij.max(dim=-1).values
                    >= args.face_gender_confidence_level
                )
            )
            .nonzero()
            .view([-1])
            .tolist()
        )
        if len(idxs_w_face_feats_from_ori) > 0:
            face_feats_1 = get_face_feats(
                face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_ori]
            )
            face_feats_target_1 = face_feats_ori_ij[idxs_w_face_feats_from_ori]
            loss_face_ij[idxs_w_face_feats_from_ori] = (
                1 - (face_feats_1 * face_feats_target_1).sum(dim=-1)
            ).to(loss_face_ij.dtype)

        images_small_ij = transforms.Resize(args.img_size_small)(images_ij)
        clip_feats_ij = get_clip_feat(
            images_small_ij, normalize=True, to_high_precision=True
        )
        loss_CLIP_ij = -(clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
        dynamic_weights = gen_dynamic_weights(
            face_indicators_ij,
            targets_ij,
            preds_gender_ori_ij,
            probs_gender_ori_ij,
            factor=args.factor1,
        )
        loss_fair.append((loss_fair_ij * 1).sum())
        loss_face.append((args.weight_loss_face * loss_face_ij).mean())
        loss_clip.append(
            (args.weight_loss_img * dynamic_weights * (loss_CLIP_ij)).mean()
        )
        # loss.append(loss_ij.mean())
    targets = targets2
    # ################################################
    #     # Step 4: compute loss

    idxs_i = list(range(targets.shape[0]))
    N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
    for j in range(N_backward):
        idxs_ij = idxs_i[
            j * args.train_GPU_batch_size : (j + 1) * args.train_GPU_batch_size
        ]

        targets_ij = targets[idxs_ij]

        preds_race_ori_ij = preds_race_ori[idxs_ij]
        probs_race_ori_ij = probs_race_ori[idxs_ij]
        face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]
        face_feats_ori_ij = face_feats_ori[idxs_ij]
        clip_feats_ori_ij = clip_feats_ori[idxs_ij]

        images_ij = images[idxs_ij]
        (
            face_indicators_ij,
            face_bboxs_ij,
            face_chips_ij,
            face_landmarks_ij,
            aligned_face_chips_ij,
        ) = get_face(images_ij)
        preds_race_ij, probs_race_ij, logits_race_ij = get_face_race(
            face_chips_ij, selector=face_indicators_ij, fill_value=-1
        )

        #

        idxs_w_face_loss = (
            ((face_indicators_ij == True) * (targets_ij != -1)).nonzero().view([-1])
        )
        loss_fair_ij_w_face_loss = CE_loss(
            logits_race_ij[idxs_w_face_loss], targets_ij[idxs_w_face_loss]
        )
        loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=device) * (0)
        loss_fair_ij[idxs_w_face_loss] = loss_fair_ij_w_face_loss

        loss_face_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=device) * (0)
        idxs_w_face_feats_from_ori = (
            (
                (face_indicators_ij == True)
                * (targets_ij != -1)
                * (targets_ij == preds_race_ori_ij)
                * (
                    probs_race_ori_ij.max(dim=-1).values
                    >= args.face_gender_confidence_level
                )
            )
            .nonzero()
            .view([-1])
            .tolist()
        )
        if len(idxs_w_face_feats_from_ori) > 0:
            face_feats_1 = get_face_feats(
                face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_ori]
            )
            face_feats_target_1 = face_feats_ori_ij[idxs_w_face_feats_from_ori]
            loss_face_ij[idxs_w_face_feats_from_ori] = (
                1 - (face_feats_1 * face_feats_target_1).sum(dim=-1)
            ).to(loss_face_ij.dtype)
        #
        images_small_ij = transforms.Resize(args.img_size_small)(images_ij)
        clip_feats_ij = get_clip_feat(
            images_small_ij, normalize=True, to_high_precision=True
        )
        loss_CLIP_ij = -(clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
        dynamic_weights = gen_dynamic_weights(
            face_indicators_ij,
            targets_ij,
            preds_race_ori_ij,
            probs_race_ori_ij,
            factor=args.factor1,
        )
        loss_fair.append((loss_fair_ij * 1).sum())
        loss_face.append((args.weight_loss_face * loss_face_ij).mean())
        loss_clip.append(
            (args.weight_loss_img * dynamic_weights * (loss_CLIP_ij)).mean()
        )

    # loss_all.backward
    fair_loss = sum(loss_fair) / len(loss_fair) / 2
    face_loss = sum(loss_face) / len(loss_face) / 2
    clip_loss = sum(loss_clip) / len(loss_clip) / 2
    num_face = abs(
        (~face_indicators).sum().item() - (~face_indicators_ori).sum().item()
    )
    # print(fair_loss.item(), clip_loss.item(), face_loss.item())
    loss = fair_loss + clip_loss + num_face

    if devide == False:

        return loss
    else:
        return fair_loss, face_loss


if __name__ == "__main__":

    device = torch.device(f"cuda:1")
    folder_path_ori = "/finetune/exp-1-debias-gender/train_outputs/gen_tensors/step_0"
    tensor = torch.load(
        "/finetune/exp-1-debias-gender/train_outputs/gen_tensors/step_0/prompt_0.pt",
        map_location=device,
    )
    pt_files = [f for f in os.listdir(folder_path_ori) if f.endswith(".pt")]
    images_tensor = []

    for i in range(len(pt_files)):
        images_tensor.append(torch.rand_like(tensor, device=device))

    for i, pt_file in tqdm(enumerate(pt_files), desc="compute loss", unit="file"):
        file_path_ori = os.path.join(folder_path_ori, pt_file)
        images = images_tensor[i]
        images_ori = torch.load(file_path_ori, map_location=device)
        if i == 0:
            fair_loss, face_loss = eval_images(images, images_ori, devide=True)
        else:
            fair_loss_value, face_loss_value = eval_images(
                images, images_ori, devide=True
            )
            fair_loss += fair_loss_value
            face_loss += face_loss_value
    print(fair_loss, face_loss)

    # for i, pt_file in tqdm(enumerate(pt_files), desc="compute loss", unit="file"):
    #     file_path_ori = os.path.join(folder_path_ori, pt_file)
    #     file_path = os.path.join(folder_path, pt_file)
    #     images = torch.load(file_path).to(device)
    #     images_ori = torch.load(file_path_ori).to(device)
    #     if i == 0:
    #         fair_loss, face_loss = eval_images(images, images_ori, devide=True)
    #     else:
    #         fair_loss_value, face_loss_value = eval_images(
    #             images, images_ori, devide=True
    #         )
    #         fair_loss += fair_loss_value
    #         face_loss += face_loss_value
    # print(fair_loss, face_loss)
