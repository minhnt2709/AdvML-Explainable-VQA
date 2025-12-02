# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

from data_utils import MyDataLoader, PromptTemplate

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

data_loader = MyDataLoader(image_path="/home/jnlp/minhnt/AdvML/custom_dataset", data_path="/home/jnlp/minhnt/AdvML/custom_dataset/dev_split.csv", split="dev")
data, image_dir = data_loader.data

image = Image.open(f"{image_dir}/{data.iloc[0]['file']}").convert("RGB")
question = f"Based on the image, analyze the question: {data.iloc[0]['question']}. Then, provide a concise answer in the following template: Answer: ."

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
