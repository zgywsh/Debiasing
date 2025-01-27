import subprocess
import torch
from torch import nn

# from eval_test import eval_images
import os
import torch
from tqdm import tqdm
from diffusers.optimization import get_scheduler


if __name__ == "__main__":

    command = [
        "python",
        "/eval-generated-images.py",
        "--generated_imgs_dir",
        "/outputs/supp/g_r",
        "--save_dir",
        "/supp/g_r/results",
    ]
    subprocess.run(command)
