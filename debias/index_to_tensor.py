import torch
import re
from diffusers.models.attention_processor import LoRAAttnProcessor


def extract_indices(file_path):
    indices = []
    index_pattern = re.compile(r"index:\s*(\d+)")

    with open(file_path, "r") as file:
        for line in file:
            match = index_pattern.search(line)
            if match:
                indices.append(int(match.group(1)))

    return indices


def saveindex():
    file_path1 = "/gender/gender/10000000_max.txt"
    file_path2 = "/gender/have gender/10000000_max.txt"
    file_path3 = "/gender/no gender/10000000_max.txt"

    list1 = extract_indices(file_path1)
    list2 = extract_indices(file_path2)
    list3 = extract_indices(file_path3)
    set1 = set(list1)
    set2 = set(list2)
    set3 = set(list3)
    index = list((set1 & set2) - set3)

    with open("gender_index.txt", "w") as file:
        for item in index:
            file.write(f"{item}\n")


def load_mask(unet, indexes, tensor):
    all_gradients = []
    original_shapes = {}

    for key, value in unet.state_dict().items():
        original_shapes[key] = value.shape
        all_gradients.append(value.view(-1))
    all_gradients = torch.cat(all_gradients)

    for i, index in enumerate(indexes):
        # print(all_gradients[index], tensor[i])
        all_gradients[index] = tensor[i]
        # print(all_gradients[index])

    start = 0
    result = {}
    for key, shape in original_shapes.items():
        size = torch.prod(torch.tensor(shape))
        result[key] = all_gradients[start : start + size].view(*shape)
        start += size
    unet.load_state_dict(result, strict=True)


def get_mask(unet, indexes):
    original_shapes = {}
    for key, value in unet.state_dict().items():
        original_shapes[key] = value.shape

    total_size = sum(value.numel() for value in unet.state_dict().values())
    all_gradients = torch.zeros(total_size, dtype=torch.float32)
    for i, index in enumerate(indexes):
        all_gradients[index] = 1

    start = 0
    result = {}
    for key, shape in original_shapes.items():
        size = torch.prod(torch.tensor(shape))
        result[key] = all_gradients[start : start + size].view(*shape)
        start += size
    return result


def get_parm(unet, indexes):
    tensor = torch.zeros(len(indexes))
    all_gradients = []
    for key, value in unet.state_dict().items():
        all_gradients.append(value.view(-1))
    all_gradients = torch.cat(all_gradients)

    for i, index in enumerate(indexes):
        tensor[i] = all_gradients[index]
        all_gradients[index] *= tensor[i]

    return tensor


def eee(unet):
    unet_lora_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        unet_lora_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=50,
        )

    unet.set_attn_processor(unet_lora_procs)

    unet_lora_dict = torch.load(
        "/unet_lora.pth",
    )
    _ = unet.load_state_dict(unet_lora_dict, strict=False)


if __name__ == "__main__":

    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        "/runwayml/sd-v1-5",
        subfolder="unet",
    )
    unet_before = UNet2DConditionModel.from_pretrained(
        "/runwayml/sd-v1-5",
        subfolder="unet",
    )
    with open("/gender_index.txt", "r") as file:

        indexes = [int(line.strip()) for line in file]
