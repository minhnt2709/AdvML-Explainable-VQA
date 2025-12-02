import ast
import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from PIL import ImageDraw
from data_utils import MyDataLoader
from tqdm import tqdm
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

task_prompt = "<DENSE_REGION_CAPTION>"
# task_prompt = "<OD>"
# text_input = "What material is the yellow cylinder that is to the right of the tiny rubber object in front of the red metal block made of?"
text_input = None
if text_input is None:
    prompt = task_prompt
else:
    prompt = task_prompt + text_input
    
data_loader = MyDataLoader(image_path="/home/jnlp/minhnt/AdvML/custom_dataset",data_path="/home/jnlp/minhnt/AdvML/custom_dataset/scene_graph_qa/dev_obj_detect_results.csv", split="dev_obj_detect")
data, image_dir = data_loader.data

res = []

for i, row in tqdm(data.iterrows(), total=len(data)):
    # if row['file'] != 'f376248cea284b39dea83c5d53ba14d9.png':
    #     continue
    img_path = os.path.join(image_dir, row['file'])
        
    # img_path = "/home/jnlp/minhnt/AdvML/custom_dataset/test/0c5b319ec09c52d00503f020ee2e3d66.png"
    image = Image.open(img_path).convert("RGB")

    # prompt = prompt + str(ast.literal_eval(row['od'])['<OD>']['bboxes'])
    # print(prompt)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    row['cap_reg'] = parsed_answer
    res.append(row)
    print(parsed_answer)

    draw = ImageDraw.Draw(image)
    boxs = parsed_answer[task_prompt]['bboxes']
    labels = parsed_answer[task_prompt]['labels']
    for box, label in zip(boxs, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}", fill="white")
        
    image.save(f"/home/jnlp/minhnt/AdvML/custom_dataset/scene_graph_qa/{row['file']}")
    
    break

# pd.DataFrame(res).to_csv("/home/jnlp/minhnt/AdvML/custom_dataset/scene_graph_qa/dev_obj_detect_results.csv", index=False)