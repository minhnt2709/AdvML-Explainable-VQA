import os
import random
import json
from turtle import pd

from PIL import Image

from transformers import AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from argparse import ArgumentParser
from dataclasses import asdict

from data_utils import MyDataLoader, PromptTemplate

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# def convert_image_mode(image: Image.Image, to_mode: str):
#     if image.mode == to_mode:
#         return image
#     elif image.mode == "RGBA" and to_mode == "RGB":
#         return rgba_to_rgb(image)
#     else:
#         return image.convert(to_mode)

# InternVL
def run_internvl(args, questions: list[str], modality: str):
    # model_name = "OpenGVLab/InternVL3-2B"

    engine_args = EngineArgs(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return {
        "engine_args": engine_args,
        "prompts": prompts,
        "stop_token_ids": stop_token_ids,
    }

# Gemma 3
def run_gemma3(args, questions: list[str], modality: str):
    assert modality == "image"
    # model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=args.model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    
    return {
        "engine_args": engine_args,
        "prompts": prompts,
    }

def run_qwen2_5_vl(args, questions: list[str], modality: str):
    # model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=args.model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    # prompts = [
    #     (
    #         "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    #         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
    #         f"{question}<|im_end|>\n"
    #         "<|im_start|>assistant\n"
    #     )
    #     for question in questions
    # ]
    
    prompts = [
        (
            f"<|im_start|>system\n{args.prompt['system']}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return {
        "engine_args": engine_args,
        "prompts": prompts,
    }
    
# Qwen3-VL-Dense
def run_qwen3_vl(args, questions: list[str], modality: str):
    # model_name = "Qwen/Qwen3-VL-4B-Instruct"

    engine_args = EngineArgs(
        model=args.model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    # prompts = [
    #     (
    #         "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    #         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
    #         f"{question}<|im_end|>\n"
    #         "<|im_start|>assistant\n"
    #     )
    #     for question in questions
    # ]
    
    prompts = [
        (
            f"<|im_start|>system\n{args.prompt['system']}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return {
        "engine_args": engine_args,
        "prompts": prompts,
    }

    
def get_multi_modal_input(args, data):
    """
    return {
        "images": list of images,
        "questions": list of questions,
    }
    """
    images = [Image.open(os.path.join(args.image_dir, image_path)).convert("RGB") for image_path in data['file']]
    # questions = [x for x in data['question']]
    
    # Input image and question
    # image = Image.open(image_path).convert("RGB")
    
    # input_instruction = "Based on the image, answer the question in one word."
    # input_instruction = "Let's think step by step. Generate your reasoning and then answer the question based on the image in one word. You MUST follow the format Explanation: <explanation>. Final Answer: <answer>"
    
    # prompt for VQA infer
    # input_prompts = [f"{input_instruction} Question: {question}" for question in questions]
    
    # prompt for scene graph generation
    input_prompts = []
    for i, row in data.iterrows():
        input_prompts.append(args.prompt['user'].format(question=row['question']))

    return {
        "images": images,
        "questions": input_prompts,
    }

def parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--image_path",
        type=str,
        default="/home/jnlp/minhnt/AdvML/custom_dataset",
        help="Path to the images",
    )
    arg_parser.add_argument(
        "--data_path",
        type=str,
        default="/home/jnlp/minhnt/AdvML/custom_dataset",
        help="Path to the dataset",
    )
    arg_parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/vqa_infer.yaml",
        help="Path to the prompt template",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to use (train_split, train_all, dev, test)",
    )
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Name of the model to use",
    )
    arg_parser.add_argument(
        "--lora_path",
        type=str,
        default="",
        help="Path to the LoRA weights",
    )
    arg_parser.add_argument(
        "--out_path",
        type=str,
        default="qwen2_5_vl_3b_dev_output.jsonl",
        help="Output file name",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return arg_parser.parse_args()

model_example_map = {
    "gemma3": run_gemma3,
    "internvl_chat": run_internvl,
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen3_vl": run_qwen3_vl,
}
        
def main():
    args = parser()
    
    if args.model_name.find("Qwen2.5-VL") != -1:
        args.model_type = "qwen2_5_vl"
    elif args.model_name.find("Qwen3-VL") != -1:
        args.model_type = "qwen3_vl"
    elif args.model_name.find("InternVL") != -1:
        args.model_type = "internvl_chat"
    elif args.model_name.find("gemma-3") != -1:
        args.model_type = "gemma3"
    
    data_loader = MyDataLoader(image_path=args.image_path, data_path=args.data_path, split=args.split)
    data, image_dir = data_loader.data
    args.image_dir = image_dir
    
    ### REMEMBER to update VQA infer prompt with this template
    prompt_template = PromptTemplate(template_path=args.prompt_path)
    args.prompt = prompt_template.get_templates()

    
    input_data = get_multi_modal_input(args, data)
    images = input_data["images"]
    questions = input_data["questions"]
    
    req_data = model_example_map[args.model_type](args, questions, modality="image")
    engine_args = asdict(req_data['engine_args'])
    prompts = req_data['prompts']
    
    print("prompt", prompts[0])
    
    llm = LLM(**engine_args)
    
    res = []
    
    # sampling params
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    assert len(prompts) == len(images)
    assert len(prompts) == len(data)
    
    if args.debug:
        data = data[:5]
        images = images[:5]
        prompts = prompts[:5]
    
    for i in range(len(data)):
        prompt = prompts[i]
        image = images[i]
        id = data.iloc[i]['id']
        img_name = data.iloc[i]['file']
        question = data.iloc[i]['question']
        # config input
        uuid = "uuid_0"
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
            "multi_modal_uuids": {"image": uuid},
        }
        
        lora_request = LoRARequest("sql_adapter", 1, args.lora_path) if len(args.lora_path) > 1 else None
        
        # llm generate
        outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0]
                
        generated_text = outputs.outputs[0].text
        
        out_sample = {
            "id": int(id),
            "file": img_name, 
            "question": question,
            "scene_graph": generated_text,
        }
        
        # out_sample = {
        #     "id": int(id),
        #     "prompt": prompt,
        #     "answer": generated_text,
        #     "explanation": "",
        # }
        res.append(out_sample)

    with open(f"{args.out_path}", "w") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")
    
if __name__ == "__main__":
    main()