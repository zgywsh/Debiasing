# !/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import pickle
from collections import Counter
import csv
import itertools
import math
import os

import cupy as cp
import time
import numpy as np
from torch.nn.utils import parameters_to_vector
import torch
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
import json
import pytz

import torch
import torch.optim as optim

from torch import nn
from torchvision import transforms
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

# os.environ["gpu_ids"] = "1"
my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.


import os
import torch
import time
from collections import Counter
import concurrent.futures


def process_file(file_path, num):
    data = cp.load(file_path)
    return data


def load_index_file(file_path):
    index_ranges = {}
    with open(file_path, "r") as file:
        for line in file:
            key, range_str = line.strip().split(": ")
            start_idx, end_idx = map(int, range_str.split(" - "))
            index_ranges[key] = (start_idx, end_idx)
    return index_ranges


def find_key_for_index(index_ranges, index):
    for key, (start, end) in index_ranges.items():
        if start <= index < end:
            return key
    return None


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
        default=1,
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
        default=8,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/outputs",
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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def global_sensitivity_analysis(
    job,
    noises,
    tokenizer,
    text_encoder,
    unet,
    vae,
    noise_scheduler,
    num_denoising_steps=30,
    guidance_scale=7.5,
    device="cuda:0",
    weight_dtype=torch.float16,
    weight_dtype_high_precision=torch.float32,
    debias="bias",
    num=10**7,
):

    optimizer = optim.Adam(unet.parameters())

    prompt_em = []

    losses = []
    times = []
    if debias == "bias":
        prompt1 = f"a photo of a male {job}"
        prompt2 = f"a photo of a female {job}"
    elif debias == "have_bias":
        prompt1 = "A male teacher is playing billiards."
        prompt2 = f"A female {job} riding a horse"
    elif debias == "no_bias":
        prompt1 = f"a photo of a old {job}"
        prompt2 = "An astronaut riding a horse"

    # race = ["black race", "yellow race", "white race"]
    # a1, a2 = random.sample(race, 2)
    # if debias == "bias":
    #     prompt1 = f"a photo of a {a1} {job}"
    #     prompt2 = f"a photo of a {a2} {job}"
    # elif debias == "have_bias":
    #     prompt1 = f"A {a1} teacher is playing billiards."
    #     prompt2 = f"A {a2} {job} riding a horse"
    # elif debias == "no_bias":
    #     prompt1 = f"a photo of a old {job}"
    #     prompt2 = "An astronaut riding a horse"

    # age = ["young", "middle-aged", "old-aged"]
    # a1, a2 = random.sample(age, 2)
    # if debias == "bias":
    #     prompt1 = f"a photo of a {a1} {job}"
    #     prompt2 = f"a photo of a {a2} {job}"
    # elif debias == "have_bias":
    #     prompt1 = f"A {a1} teacher is playing billiards."
    #     prompt2 = f"A {a2} {job} riding a horse"
    # elif debias == "no_bias":
    #     prompt1 = f"a photo of a male {job}"
    #     prompt2 = "An astronaut riding a horse"

    for n, prompt in enumerate((prompt1, prompt2)):
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
        prompt_em.append(prompt_embeds)

    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    latents1 = noises
    latents2 = noises

    for i, t in enumerate(noise_scheduler.timesteps):
        # scale model input
        latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        latent_model_input1 = latent_model_input
        latent_model_input2 = latent_model_input

        noises_pred1 = unet(
            latent_model_input1,
            t,
            encoder_hidden_states=prompt_em[0],
        ).sample
        noises_pred2 = unet(
            latent_model_input2,
            t,
            encoder_hidden_states=prompt_em[1],
        ).sample

        # Assuming differences are computed correctly and requires_grad=True
        loss = torch.sum(torch.abs(noises_pred1 - noises_pred2))
        losses.append(loss.item())
        times.append(t.item())  #

        optimizer.zero_grad()  #
        #
        loss.backward(retain_graph=False)

        grad_dict = {}
        for name, param in unet.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.detach()
        if t == 699:
            all_gradients = []
            for value in grad_dict.values():
                if isinstance(value, torch.Tensor):
                    all_gradients.append(value.abs().view(-1))
            if all_gradients:
                all_gradients = torch.cat(all_gradients)
                _, indices = torch.topk(
                    all_gradients, k=min(num, all_gradients.numel()), largest=True
                )
                # index = cp.array(indices.cpu().numpy())
                cp.save(
                    f"/outputs/20jobs/grad/{debias}/{num}_{job}",
                    indices,
                )
            continue

        noises_pred1 = noises_pred1.to(weight_dtype_high_precision)
        noises_pred_uncond1, noises_pred_text1 = noises_pred1.chunk(2)
        noises_pred1 = noises_pred_uncond1 + guidance_scale * (
            noises_pred_text1 - noises_pred_uncond1
        )
        latents1 = noise_scheduler.step(noises_pred1, t, latents1).prev_sample

        noises_pred2 = noises_pred2.to(weight_dtype_high_precision)
        noises_pred_uncond2, noises_pred_text2 = noises_pred2.chunk(2)
        noises_pred2 = noises_pred_uncond2 + guidance_scale * (
            noises_pred_text2 - noises_pred_uncond2
        )
        latents2 = noise_scheduler.step(noises_pred2, t, latents2).prev_sample


def gen_grad(args, num):
    args.device = f"cuda:{args.gpu_id}"

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    text_encoder.to(args.device, dtype=weight_dtype)
    unet.to(args.device, dtype=weight_dtype)
    vae.to(args.device, dtype=weight_dtype)

    # Dataset and DataLoaders creation:
    with open(args.prompts_path, "r") as f:
        experiment_data = json.load(f)
    jobs = experiment_data["jobs"]

    noise = torch.randn([1, 4, 64, 64], dtype=weight_dtype_high_precision).to(
        args.device
    )
    for debias in ("no_bias", "bias", "have_bias"):
        for i, jo in enumerate(jobs):
            print(i, jo)
            global_sensitivity_analysis(
                job=jo,
                noises=noise,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                vae=vae,
                noise_scheduler=noise_scheduler,
                num_denoising_steps=args.num_denoising_steps,
                guidance_scale=args.guidance_scale,
                device=args.device,
                weight_dtype=weight_dtype,
                weight_dtype_high_precision=weight_dtype_high_precision,
                debias=debias,
                num=num,
            )


def count_frequencies_in_folder(fre):

    for debias in ("no_bias", "bias", "have_bias"):
        total_counts = cp.zeros(9 * (10**8), dtype=cp.int16)
        folder_path = f"/outputs/20jobs/grad/{debias}/"

        filenames = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

        for filename in tqdm(filenames, desc="Processing files", unit="file"):
            file_path = os.path.join(folder_path, filename)
            data = cp.load(file_path)
            unique, counts = cp.unique(data, return_counts=True)
            #
            total_counts[unique] += counts

        indices_greater_than_n = cp.where(total_counts > fre * len(filenames))[0]
        # print(len(indices_greater_than_n))

        # #
        cp.save(
            f"/outputs/20jobs/index/gender/{debias}",
            indices_greater_than_n,
        )
    return


def gen_index(num):
    for debias in ("bias", "have_bias"):
        start_time = time.time()
        all_indices = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            # Iterate over files in the directory
            with os.scandir(f"/outputs/20jobs/{debias}/") as it:
                for entry in it:
                    if entry.name.endswith(".npy") and entry.is_file():
                        futures.append(executor.submit(process_file, entry.path, num))

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                all_indices.extend(future.result())
        #
        file_path = f"/outputs/20jobs/{debias}/{num}_indices.pkl"
        #
        cp.save(file_path, all_indices)
        #
        all_indices = cp.load(file_path)

        print("stat_computing")
        #
        array = all_indices
        #
        unique, counts = cp.unique(array, return_counts=True)
        frequency = dict(zip(unique.get(), counts.get()))

        # Filtering indices with frequency greater than 40
        filtered_indices = [
            (index, freq) for index, freq in frequency.items() if freq > 40
        ]

        # Loading index ranges from file
        index_ranges = load_index_file("/gradient_info.txt")

        # Printing results
        with open(f"/outputs/{debias}/{num}_max.txt", "w") as file:
            for rank, (index, freq) in enumerate(filtered_indices, start=1):
                key = find_key_for_index(index_ranges, index)
                print(f"top_{rank}, index:{index}, freq:{freq}, key: {key}")
                file.write(f"top_{rank}, index:{index}, freq:{freq}, key: {key}\n")
        end_time = time.time()
        print(f"compute_time: {end_time - start_time} s")


def save_index(folder_path):
    #
    data1 = np.load(os.path.join(folder_path, "bias.npy"))
    data2 = np.load(os.path.join(folder_path, "have_bias.npy"))
    data3 = np.load(os.path.join(folder_path, "no_bias.npy"))
    print(len(data1), len(data2), len(data3))

    #
    intersection = np.intersect1d(data1, data2)

    #
    result = np.setdiff1d(intersection, data3)
    print(len(result))

    output_file_path = os.path.join(folder_path, "result.txt")
    np.savetxt(output_file_path, result, fmt="%s")


if __name__ == "__main__":
    args = parse_args()

    num = 10**7
    # gen_grad(args, num)
    count_frequencies_in_folder(0.65)
    save_index("/outputs/20jobs/index/gender")


####loss
# import os
# import csv
# import matplotlib.pyplot as plt
#

# folder_path = 'outputs/loss'
#

# data = {}
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         filepath = os.path.join(folder_path, filename)
#         with open(filepath, mode='r') as file:
#             reader = csv.DictReader(file)
#             times = []
#             losses = []
#             for row in reader:
#                 times.append(float(row['Time Step']))
#                 losses.append(float(row['Loss']))
#             data[filename] = (times, losses)
#
