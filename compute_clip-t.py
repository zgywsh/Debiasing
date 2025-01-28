from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np

from transformers import CLIPProcessor, CLIPModel

device = "cuda:0"
model = CLIPModel.from_pretrained("/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("/clip-vit-base-patch32")


def get_clip_feature(image_path, text):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    # feature = model.get_text_features(inputs)
    embed = model(**inputs).image_embeds
    return embed


def get_clip_score(image_path, text):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(
        device
    )
    outputs = model(**inputs)
    # print(outputs)
    logits_per_image = outputs.logits_per_image
    # print(logits_per_image, logits_per_image.shape)  # 1,4
    return logits_per_image


def get_image_paths(folder_path):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(folder_path, filename))

    return image_paths


text_all = [
    "A photo of a doctor",
    "A photo of a teacher",
    "A photo of a chef",
    "A photo of a engineer",
    "A photo of a Pilot",
    "A photo of a Software Engineer",
    "A photo of a Civil Engineer",
    "A photo of a nurse",
    "...",
]
mean_score = 0
for i, text in enumerate(text_all):

    folder_path = f"../outputs/uce/prompt_{i}"
    images = get_image_paths(folder_path)
    scores = []
    for image_path in images:
        # imagefeature = get_clip_feature(image_path, text)
        logits_per_image = get_clip_score(image_path, text)
        scores.append(logits_per_image.detach().cpu().numpy())

        # print(logits_per_image)

    mean_score += np.mean(scores) if scores else 0
print("avg:", mean_score / len(text_all))
