# X-VQA for Final Advanced Machine Learning course

main env: physicai python=3.12, vllm=0.11, transformers=4.57.3

Folder InternVL 
+ follow InternVL guidelines here https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html
+ for LoRA finetuning InternVL family
+ env internvl python=3.9, transformers=4.37.2

Folder tbd-nets
+ clone the repository https://github.com/davidmascharka/tbd-nets
+ set up the required environment, download pre-trained checkpoints
+ perform zero-shot QA using full-vqa-example.ipynb

File src/lora_ft.py
+ for LoRA finetuning PEFT trl VLM models

Folder ft_out
+ for logging LoRA Qwen ckpt

## TODO
+ [x] LoRA QwenVL, InternVL
+ [ ] Few-shot, CoT
+ [x] Scene graph: object detection / Region captioning
+ [x] Ensemble
+ [x] Two-stage TbD-net + fine-tuned VLM

