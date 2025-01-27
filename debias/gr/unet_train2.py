import argparse
import hashlib
import torch.optim as optim
from tools import find_param
from torch.optim.lr_scheduler import LambdaLR
from index_to_tensor import get_mask, eee
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
import itertools
import json
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# from insightface.app import FaceAnalysis
import warnings
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
import warnings


warnings.filterwarnings("ignore")

my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Script to finetune Stable Diffusion for debiasing purposes."
    )

    parser.add_argument(
        "--classifier_weight_path",
        default="finetune/data/2-trained-classifiers/CelebA_MobileNetLarge_08060852/epoch=9-step=12660_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender",
        type=str,
        required=False,
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
        default=24,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/finetune/exp-1-debias-gender/train_outputs/gen_images/images_debias",
        # required=True
    )
    parser.add_argument(
        "--save_dir_ori",
        type=str,
        default="/finetune/exp-1-debias-gender/train_outputs/gen_images/images_ori",
        # required=True
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=14,
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
        "--batch_size", help="batch size for image generation", type=int, default=1
    )
    parser.add_argument(
        "--EMA_decay", help="decay coefficient for EMA", type=float, default=0.996
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


args = parse_args()

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="scheduler"
)

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet"
)


def generate_image(
    prompt,
    noises,
    tokenizer,
    text_encoder,
    unet,
    unet_w,
    vae,
    noise_scheduler,
    num_denoising_steps=30,
    guidance_scale=7.5,
    device="cuda:0",
    weight_dtype=torch.float16,
    weight_dtype_high_precision=torch.float32,
    step=args.num_denoising_steps - 1,
):
    N = noises.shape[0]
    prompts = [prompt] * N

    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)

    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    for i, t in enumerate(noise_scheduler.timesteps):
        if i >= step:
            # scale model input
            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            # print(latent_model_input.shape)
            noises_pred = unet_w(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)

            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + guidance_scale * (
                noises_pred_text - noises_pred_uncond
            )

            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample
        else:
            # scale model input
            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            # print(latent_model_input.shape)
            noises_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)

            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + guidance_scale * (
                noises_pred_text - noises_pred_uncond
            )

            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)  # in range [-1,1]

    return images


def gen_tensor(unet_w, rank, args, prompt_i, step):
    """unet,unet_w(19) to gen_tensor
    step in noise
    """

    torch.cuda.set_device(rank)

    unet = copy.deepcopy(unet_w)
    ######################
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    unet_w.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    images_tensor = []

    save_dir_prompt_i = os.path.join(args.save_dir, f"prompt_{i}")
    os.makedirs(save_dir_prompt_i, exist_ok=True)

    noises_to_use = []
    img_save_paths_to_use = []
    for j in range(args.num_imgs_per_prompt):
        img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")
        # if not os.path.exists(img_save_path):
        noises_to_use.append(noise_all[i, j].unsqueeze(dim=0))
        img_save_paths_to_use.append(img_save_path)
    noises_to_use = torch.cat(noises_to_use).to(device)
    start_index = step * part_size
    end_index = (step + 1) * part_size
    noises_to_use = noises_to_use[start_index:end_index]

    N = math.ceil(noises_to_use.shape[0] / args.batch_size)
    images = []
    for j in range(N):
        noises_ij = noises_to_use[args.batch_size * j : args.batch_size * (j + 1)]
        img_save_paths_ij = img_save_paths_to_use[
            args.batch_size * j : args.batch_size * (j + 1)
        ]

        images_ij = generate_image(
            prompt_i,
            noises_ij,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            unet_w=unet_w,
            vae=vae,
            noise_scheduler=noise_scheduler,
            num_denoising_steps=args.num_denoising_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            weight_dtype=torch.float16,
            weight_dtype_high_precision=torch.float32,
        )
        images.append(images_ij)
    images_tensor.append(torch.cat(images))

    return images_tensor


def gen_tensor_w(unet_w, rank, args, prompt_i):
    """uet_w gen_tensor_w"""

    torch.cuda.set_device(rank)

    unet = copy.deepcopy(unet_w)
    ######################
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    images_tensor = []

    save_dir_prompt_i = os.path.join(args.save_dir, f"prompt_{i}")
    os.makedirs(save_dir_prompt_i, exist_ok=True)

    noises_to_use = []
    img_save_paths_to_use = []
    for j in range(args.num_imgs_per_prompt):
        img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")

        # if not os.path.exists(img_save_path):
        noises_to_use.append(noise_all[i, j].unsqueeze(dim=0))
        img_save_paths_to_use.append(img_save_path)
    noises_to_use = torch.cat(noises_to_use).to(device)

    N = math.ceil(noises_to_use.shape[0] / args.batch_size)
    images = []
    for j in range(N):
        noises_ij = noises_to_use[args.batch_size * j : args.batch_size * (j + 1)]
        img_save_paths_ij = img_save_paths_to_use[
            args.batch_size * j : args.batch_size * (j + 1)
        ]

        images_ij = generate_image(
            prompt_i,
            noises_ij,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            unet_w=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            num_denoising_steps=args.num_denoising_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            weight_dtype=torch.float16,
            weight_dtype_high_precision=torch.float32,
        )
        images.append(images_ij)
    images_tensor.append(torch.cat(images))
    images_tensor = torch.cat(images_tensor)

    return images_tensor


def gen_ori(rank, args):
    """gen_tensor_ori,gen_img_ori"""

    torch.cuda.set_device(rank)

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    with open(args.prompts_path, "r") as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["train_prompts"]
    prompts_to_process = test_prompts

    noise_all = []

    for prompt in prompts_to_process:
        hash_seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000000
        # hash_seed = hash(prompt)
        noise_per_prompt = []
        for i in range(args.num_imgs_per_prompt):
            torch.manual_seed(args.random_seed + hash_seed + i)
            noise_single = torch.randn([1, 4, 64, 64], dtype=torch.float32).to(device)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    noise_all = torch.cat(noise_all)

    total_prompts = len(prompts_to_process)

    for i, prompt_i in enumerate(prompts_to_process):

        save_dir_prompt_i = os.path.join(args.save_dir_ori, f"prompt_{i}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        noises_to_use = []
        img_save_paths_to_use = []
        for j in range(args.num_imgs_per_prompt):
            img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")

            noises_to_use.append(noise_all[i, j].unsqueeze(dim=0))
            img_save_paths_to_use.append(img_save_path)
        noises_to_use = torch.cat(noises_to_use)

        N = math.ceil(noises_to_use.shape[0] / args.batch_size)
        images = []
        for j in range(N):
            noises_ij = noises_to_use[args.batch_size * j : args.batch_size * (j + 1)]
            img_save_paths_ij = img_save_paths_to_use[
                args.batch_size * j : args.batch_size * (j + 1)
            ]

            images_ij = generate_image(
                prompt_i,
                noises_ij,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                unet_w=unet,
                vae=vae,
                noise_scheduler=noise_scheduler,
                num_denoising_steps=args.num_denoising_steps,
                guidance_scale=args.guidance_scale,
                device=device,
                weight_dtype=torch.float16,
                weight_dtype_high_precision=torch.float32,
            )
            images.append(images_ij)
            # save images
            for img, img_save_path in itertools.zip_longest(
                images_ij, img_save_paths_ij
            ):
                img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)
                img_pil.save(img_save_path)

        images_tensor = torch.cat(images)

        torch.save(
            images_tensor,
            f"/finetune/exp-1-debias-gender/train_outputs/gen_tensors/step_0/{prompt_i}.pt",
        )
        print(f"save {prompt_i} success")
        # print(f"save prompt_{start_idx + i} success")

    return images_tensor


def gen_images(rank, args, step):
    """unet,unet_w(step),gen_images"""
    torch.cuda.set_device(rank)

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    unet_w = copy.deepcopy(unet)
    pretrained_weights_path = (
        "/finetune/exp-1-debias-gender/train_outputs/unet/unet_weights.pth"
    )
    unet_w.load_state_dict(torch.load(pretrained_weights_path))
    unet_w.requires_grad_(False)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    with open(args.prompts_path, "r") as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["train_prompts"]
    prompts_to_process = test_prompts

    noise_all = []

    for prompt in prompts_to_process:
        hash_seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000000
        # hash_seed = hash(prompt)
        noise_per_prompt = []
        for i in range(args.num_imgs_per_prompt):
            torch.manual_seed(args.random_seed + hash_seed + i)
            noise_single = torch.randn([1, 4, 64, 64], dtype=torch.float32).to(device)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    noise_all = torch.cat(noise_all)

    total_prompts = len(prompts_to_process)

    for i, prompt_i in enumerate(prompts_to_process):

        save_dir_prompt_i = os.path.join(args.save_dir, f"prompt_{i}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        noises_to_use = []
        img_save_paths_to_use = []
        for j in range(args.num_imgs_per_prompt):
            img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")
            # if not os.path.exists(img_save_path):
            noises_to_use.append(noise_all[i, j].unsqueeze(dim=0))
            img_save_paths_to_use.append(img_save_path)
        noises_to_use = torch.cat(noises_to_use)

        N = math.ceil(noises_to_use.shape[0] / args.batch_size)
        images = []
        for j in range(N):
            noises_ij = noises_to_use[args.batch_size * j : args.batch_size * (j + 1)]
            img_save_paths_ij = img_save_paths_to_use[
                args.batch_size * j : args.batch_size * (j + 1)
            ]

            images_ij = generate_image(
                prompt_i,
                noises_ij,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                unet_w=unet_w,
                vae=vae,
                noise_scheduler=noise_scheduler,
                num_denoising_steps=args.num_denoising_steps,
                guidance_scale=args.guidance_scale,
                device=device,
                weight_dtype=torch.float16,
                weight_dtype_high_precision=torch.float32,
                step=step,
            )
            images.append(images_ij)
            # save images
            for img, img_save_path in itertools.zip_longest(
                images_ij, img_save_paths_ij
            ):
                img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)
                img_pil.save(img_save_path)

        # print(f"save prompt_{start_idx + i} success")

    return


def unet_gen_images(step):
    args = parse_args()
    gen_images(7, args, step)
    print("Generating_ok")
    return


class CustomLRScheduler:
    def __init__(
        self, optimizer, warmup_steps, decay_steps, decay_multiplier, initial_lr
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_multiplier = decay_multiplier
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.initial_lr
        else:

            decay_factor = self.decay_multiplier ** (
                (self.current_step - self.warmup_steps) // self.decay_steps
            )
            return self.initial_lr * decay_factor


if __name__ == "__main__":
    args = parse_args()
    gpu = 7
    # torch.cuda.set_device(1)
    from unet_loss2 import eval_images

    # images_tensor_ori = gen_ori(rank=gpu, args=args)
    # print("Generating_ok")

    unet_w = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    unet_w.requires_grad_(False)

    params_train1 = find_param("/gender_index.txt")
    params_train2 = find_param("/outputs/1000jobs/index/race/race_index.txt")
    params_train = params_train1 + params_train1
    print(len(params_train))

    with open("/result.txt", "r") as file:
        indexes1 = [int(line.strip()) for line in file]
    with open("/outputs/1000jobs/index/race/race_index.txt", "r") as file:
        indexes2 = [int(line.strip()) for line in file]
    indexes = list(set(indexes2).union(set(indexes1)))
    grad_mask = get_mask(unet_w, indexes)
    for name, param in unet_w.named_parameters():
        if name in params_train:
            param.requires_grad = True

    ###############################################################
    with open(args.prompts_path, "r") as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["train_prompts"]
    prompts_to_process = test_prompts
    noise_all = []
    for prompt in prompts_to_process:
        hash_seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000000
        # hash_seed = hash(prompt)
        noise_per_prompt = []
        for i in range(args.num_imgs_per_prompt):
            torch.manual_seed(args.random_seed + hash_seed + i)
            noise_single = torch.randn([1, 4, 64, 64], dtype=torch.float32)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    noise_all = torch.cat(noise_all)
    ####################################################

    total_epochs = 50
    warmup_steps = 10
    decay_steps = 10
    decay_multiplier = 0.2
    initial_lr = 0.0001

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, unet_w.parameters()),
        lr=initial_lr,
        momentum=0.9,
    )
    scheduler = CustomLRScheduler(
        optimizer, warmup_steps, decay_steps, decay_multiplier, initial_lr
    )

    for epoch in range(total_epochs):
        loss = 0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{total_epochs}, Learning Rate: {current_lr}")
        for i, prompt_i in enumerate(prompts_to_process):
            # print(prompt_i)
            images_ori_path = f"/finetune/exp-1-debias-gender/train_outputs/gen_tensors/step_0/{prompt_i}.pt"
            images_ori_tensor = torch.load(images_ori_path, map_location=f"cuda:{gpu}")
            part_size = 4
            num_parts = int(args.num_imgs_per_prompt / part_size)
            images_tensor_w = gen_tensor_w(
                unet_w, rank=gpu, args=args, prompt_i=prompt_i
            )
            targets1, targets2 = eval_images(
                images_tensor_w, images_ori_tensor, test=True
            )
            for j in tqdm(range(num_parts), desc="Processing", unit="part"):
                optimizer.zero_grad()
                start_index = j * part_size
                end_index = (j + 1) * part_size
                images_ori = images_ori_tensor[start_index:end_index]
                target1 = targets1[start_index:end_index]
                target2 = targets2[start_index:end_index]
                images_tensor = gen_tensor(
                    unet_w, rank=gpu, args=args, prompt_i=prompt_i, step=j
                )
                images = torch.cat(images_tensor)
                true_loss = eval_images(
                    images, images_ori, iftargets=(target1, target2)
                )

                # print(f"loss: {true_loss}")

                true_loss.backward()

                for name, param in unet_w.named_parameters():
                    mask = grad_mask.get(name).to(f"cuda:{gpu}")

                    if mask is not None:

                        mask = torch.tensor(mask, dtype=torch.float32)

                        #
                        if param.grad is not None:
                            param.grad *= mask

                optimizer.step()  #
                loss += true_loss
        print(f"all_loss: {loss/i+1}")
        #
        scheduler.step()
        if epoch == 0:
            min_loss = loss
        elif loss < min_loss and epoch >= 3:
            min_loss = loss

            torch.save(
                unet_w.state_dict(),
                f"/finetune/exp-1-debias-gender/train_outputs/unet/unet_weights.pth",
            )
            print("saving")
    unet_gen_images(-1)

    #     print(f"Epoch {epoch+1}, Loss: {true_loss.item()}")
