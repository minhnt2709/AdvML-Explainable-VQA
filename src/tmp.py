import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO
from data_utils import MyDataLoader
from tqdm import tqdm
import json

IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)

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

def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
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

def load_image(image_data, input_size=384, max_num=10):
    image = Image.open(image_data).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model_path = 'LiAutoAD/Ristretto-3B'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

data_loader = MyDataLoader(image_path="/home/jnlp/minhnt/AdvML/custom_dataset",data_path="/home/jnlp/minhnt/AdvML/custom_dataset/test.csv", split="test")
data, image_dir = data_loader.data

# image_url = 'https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524'
# response = requests.get(image_url)
# image_data = BytesIO(response.content)
res = []

for i, row in tqdm(data.iterrows()):
    image_data = f"{image_dir}/{row['file']}"
    pixel_values = load_image(image_data, max_num=10).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # The recommended range for `num_image_token` is 64 to 576, and the value can be adjusted based on task requirements.
    num_image_token = 256

    # text-image conversation && multi-round conversation
    question = f'<image>         Based on the image, answer the question in one word. {row["question"]}'
    # question = f'<image>         Based on the image, answer the question and generate your reasoning. Finally, conclude the answer with only one word. Question: {row["question"]}'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    print(f'User: {question}         Assistant: {response}')
    res.append({
        "id": int(row['id']),
        "file": row['file'],
        "question": row['question'],
        "answer": response,
        "explanation": "tmp"
    })
    
with open('/home/jnlp/minhnt/AdvML/results/test/ristretto_3b_test_output_hf.jsonl', 'w') as f:
    for item in res:
        f.write(json.dumps(item) + "\n")