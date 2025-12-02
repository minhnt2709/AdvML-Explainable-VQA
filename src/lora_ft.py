import json
import torch
import ast
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from data_utils import MyDataLoader, PromptTemplate
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def format_data(sample):
    system_message = "You are a Vision Language Model that helps people answer questions based on the image provided."
    return {
      "images": [sample["image"]],
      "messages": [
          {
              "role": "system",
              "content": [
                  {
                      "type": "text",
                      "text": system_message
                  }
              ],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                      "text": f"{sample['question']} Answer: {sample['answer']}.",
                  }
              ],
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                    #   "text": sample["answer"] + ". Explanation: " + ast.literal_eval(sample["explanation"])[0],
                    # "text": f"{ast.literal_eval(sample["explanation"])[0]} Therefore the answer is {sample["answer"]}.",
                    "text": f"Explanation: {sample['explanation']}",
                  }
              ],
          },
      ]
      }
    
def format_data_test(sample, prompt):
    system_message = "You are a Vision Language Model that helps people answer questions based on the image provided."
    input_prompt = prompt['user'].format(question=sample['question'], answer=sample['answer'])
    return {
      "images": [sample["image"]],
      "messages": [

          {
              "role": "system",
              "content": [
                  {
                      "type": "text",
                      "text": system_message
                  }
              ],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                    #   "text": input_prompt,
                    "text": f"{sample['question']} Answer: {sample['answer']}.",
                  }
              ],
          }
      ]
      }
    
def prepare_dataset(data_path: str, mode='train', prompt=None, qa_path=None, image_dir=None):
    # add image column
    def add_image_column(example):
        from PIL import Image
        image_path = f"{image_dir}/{example['file']}"
        example['image'] = Image.open(image_path).convert("RGB")
        return example
    
    image_dir = image_dir if image_dir is not None else f"{data_path}/train/"
    
    if mode == 'infer_dev':
        dataset = load_dataset('csv', data_files={"dev": qa_path if qa_path is not None else f'{data_path}/dev_split.csv'})
        dev_dataset = dataset["dev"].map(add_image_column)
        dev_dataset = [format_data_test(sample, prompt) for sample in dev_dataset]
        return dev_dataset
    
    elif mode == 'infer_dev_scene_graph':
        dataset = load_dataset('csv', data_files={"dev": f'{data_path}/scene_graph_qa/dev_scene_graph/qwen3_vl_4b_dev.csv'})
        dev_dataset = dataset["dev"].map(add_image_column)
        dev_dataset = [format_data_test(sample, prompt) for sample in dev_dataset]
        return dev_dataset
    
    if mode == 'test':
        image_dir = f"{data_path}/test"
        dataset = load_dataset('csv', data_files={"test": f'{qa_path}'})
        test_dataset = dataset["test"].map(add_image_column)
        test_dataset = [format_data_test(sample, prompt) for sample in test_dataset]
        return test_dataset
    # train and dev dataset
    else:
        dataset = load_dataset('csv', data_files={"train_split": f'{data_path}/train_split.csv', "dev": f'{data_path}/dev_split.csv'})
    
        train_dataset = dataset["train_split"].map(add_image_column)
        dev_dataset = dataset["dev"].map(add_image_column)
        
        # train_dataset = [format_data(sample) for sample in train_dataset]
        # dev_dataset = [format_data(sample) for sample in dev_dataset]
        
        train_dataset_x = []
        for sample in train_dataset:
            explanations = ast.literal_eval(sample["explanation"])
            # question = sample["question"]
            # answer = sample["answer"]
            for e in explanations:
                new_sample = sample.copy()
                new_sample["explanation"] = e
                train_dataset_x.append(format_data(new_sample))
                
        dev_dataset_x = []
        for sample in dev_dataset:
            explanations = ast.literal_eval(sample["explanation"])
            # question = sample["question"]
            # answer = sample["answer"]
            for e in explanations:
                new_sample = sample.copy()
                new_sample["explanation"] = e
                dev_dataset_x.append(format_data(new_sample))
                break
        
        return train_dataset_x, dev_dataset_x

def qwenvl_generate_text_from_sample(model, processor, sample, max_new_tokens=512, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample['messages'])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

def gemma_generate_text_from_sample(model, processor, sample, max_new_tokens=512, device="cuda"):
    # Prepare the text input by applying the chat template
    print("Preparing text input...", sample['messages'][1:2])
    
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = text_input["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**text_input, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    output_text = processor.decode(generation, skip_special_tokens=True)

    return output_text[0]  # Return the first decoded output text

def lora_train():
    train_dataset, dev_dataset = prepare_dataset(data_path="/home/jnlp/minhnt/AdvML/custom_dataset")
    print(len(train_dataset), len(dev_dataset))
    
    # return 1

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    if model_id.find("Qwen2.5-VL") != -1:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_id)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="ft_out/Qwen2.5-VL-3B-instruct-trl-sft-19k-2eps",  # Directory to save the model
        num_train_epochs=2,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        max_length=None,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        # Logging and evaluation
        logging_steps=100,  # Steps interval for logging
        eval_steps=100,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=100,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        # push_to_hub=True,  # Whether to push model to Hugging Face Hub
        # report_to="trackio",  # Reporting tool for tracking metrics
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
def lora_infer():
    prompt = PromptTemplate("prompts/x_with_ans_gen.yaml").get_templates()
    
    split = "infer_dev"
    
    test_dataset = prepare_dataset(data_path="/home/jnlp/minhnt/AdvML/custom_dataset", mode=split, prompt=prompt, qa_path=f"/home/jnlp/minhnt/AdvML/custom_dataset/dev_split.csv", image_dir=None)
    # print(test_dataset[0])

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # model_id = "google/gemma-3-4b-it"

    if model_id.find("Qwen2.5-VL") != -1:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        generate_text_from_sample = qwenvl_generate_text_from_sample
        
    elif model_id.find("gemma") != -1:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="cuda", torch_dtype=torch.bfloat16,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        generate_text_from_sample = gemma_generate_text_from_sample
    
    # adapter_path = "/home/jnlp/minhnt/AdvML/ft_out/qwen25-vl-3b-instruct-trl-sft"
    # adapter_path = "/home/jnlp/minhnt/AdvML/ft_out/Qwen2.5-VL-3B-instruct-trl-sft-v3"
    adapter_path = "/home/jnlp/minhnt/AdvML/ft_out/Qwen2.5-VL-3B-instruct-trl-sft-19k-2eps"
    model.load_adapter(adapter_path)
    
    
    res = []
    for sample in tqdm(test_dataset):        
        print(sample)
        output = generate_text_from_sample(model, processor, sample)
        res.append({
            "question": sample["messages"][1]['content'][1]['text'],
            "answer": "tmp",
            "explanation": output,
        })
        print("Output:", output)
        # break
        
    with open(f"results/dev_gen_x/qwen25-vl-3b-instruct-trl-sft_19k-2eps_genx.jsonl", "w") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")
    
    
if __name__ == "__main__":
    # lora_train()
    lora_infer()