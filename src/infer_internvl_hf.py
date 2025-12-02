import numpy as np
import torch
import torchvision.transforms as T
import json
import os
import pandas as pd

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from data_utils import MyDataLoader
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
# path = 'OpenGVLab/InternVL2_5-2B'
# path = "/home/jnlp/minhnt/AdvML/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_4b_dynamic_res_2nd_finetune_lora_merge"
# path = "/home/jnlp/minhnt/AdvML/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_4b_dynamic_res_2nd_finetune_lora_unfreezeViT_2eps_merge"

# 11k samples for explanation generation
path = "/home/jnlp/minhnt/AdvML/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_4b_dynamic_res_2nd_finetune_lora_gen_x_merge"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
split = "test"
data_loader = MyDataLoader(image_path="/home/jnlp/minhnt/AdvML/custom_dataset", data_path="/home/jnlp/minhnt/AdvML/custom_dataset/test.csv", split=split)
data, image_dir = data_loader.data

res = []

for i, row in tqdm(data.iterrows()):
    # if i == 20:
    #     break
    if split == "test":
        test_tbd = pd.read_csv('/home/jnlp/minhnt/AdvML/results/test_gen_x/tbd_nets.csv')
        row['answer'] = test_tbd.loc[test_tbd['id'] == int(row['id']), 'answer'].values[0]
        # print(f"ID: {row['id']}, Answer: {row['answer']}")
        # break
    image_data = f"{image_dir}/{row['file']}"
    pixel_values = load_image(image_data, max_num=10).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=100, do_sample=True)

    # instruction for explanation generation
    # question = f'<image>         Based on the input image, question, and answer, generate an explanation. Question: {row["question"]} Answer: {row["answer"]}. Explanation:'
    # question = f'<image>         {row["question"]} Answer: {row["answer"]}. Explanation:'
    question = f'<image>         {row["question"]} Answer: {row["answer"]}.'
    
    # question = f'<image>         Based on the image, answer the question in one word. {row["question"]}'
    # question = f'<image>         Based on the image, answer the question and generate your reasoning. Finally, conclude the answer with only one word. Question: {row["question"]}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}         Assistant: {response}')
    if split == "test":
        res.append({
            "id": int(row['id']),
            "answer": row["answer"],
            "explanation": response
        })
    else:
        res.append({
            "id": int(row['id']),
            "file": row['file'],
            "question": row['question'],
            "answer": row["answer"],
            "explanation": response
        })
    
    print(response)
    # break
    
with open('/home/jnlp/minhnt/AdvML/results/test_gen_x/internvl2_5_4b_sft_11k_genx_v3.jsonl', 'w') as f:
    for item in res:
        f.write(json.dumps(item) + "\n")