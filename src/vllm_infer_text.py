import os
import random
import json

from PIL import Image

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from argparse import ArgumentParser

from data_utils import MyDataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2,
    )
    llm = LLM(model_id, gpu_memory_utilization=0.8)
    
    data_loader = MyDataLoader(image_path="/home/jnlp/minhnt/AdvML/custom_dataset", split="test", data_path="/home/jnlp/minhnt/AdvML/custom_dataset/test.csv")
    data, image_dir = data_loader.data
    
    res = []
    
    for _, sample in data.iterrows():
        id = sample['id']
        img_name = sample['file']
        question = sample['question']
        # answer = sample['answer']
        explanation = ""
        
        # llm generate
        # input = f"Question: {question} What is the questin about? Select from [binary question, color, shape, material, count, size]. Answer only with ONE of these options."
        input = f"""You MUST choose ONE option from [binary, color, shape, material, count, size] that best describes what the question is about. Do not answer with anything else.
        Example 1:
        Question: There is a tiny red thing that is right of the brown cylinder behind the small rubber sphere; are there any tiny red balls on the right side of it? 
        Answer: binary
        Example 2:
        Question: There is a block in front of the cube that is behind the object that is in front of the small purple metallic thing; what size is it?
        Answer: size
        Example 3:
        Question: The large metal object that is the same shape as the tiny purple thing is what color?
        Answer: color
        Example 4: What is the big red block made of?
        Answer: material
        Example 5: How many other objects are there of the same shape as the large brown matte object?
        Answer: count
        Example 6: What is the shape of the large object that is the same color as the tiny metallic ball?
        Answer: shape
        
        Question: {question}
        Answer:"""
        outputs = llm.generate(input, sampling_params=sampling_params)
                
        generated_text = outputs[0].outputs[0].text
        
        print(generated_text)
        
        sample['question_type'] = generated_text.strip()
        res.append(sample.to_dict())
        
        # break
    
    with open("/home/jnlp/minhnt/AdvML/custom_dataset/test_question_type.jsonl", "w") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")
    
if __name__ == "__main__":
    main()