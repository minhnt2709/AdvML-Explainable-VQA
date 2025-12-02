from datasets import load_dataset
from transformers import (
    ViltProcessor,
    DefaultDataCollator,
    ViltForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import torch
from PIL import Image

import pandas as pd

# datafiles = {
#     "train": "/home/jnlp/minhnt/AdvML/custom_dataset/train_split.csv",
#     "validation": "/home/jnlp/minhnt/AdvML/custom_dataset/dev_split.csv",
# }
# dataset = load_dataset("csv", data_files=datafiles)
dataset = load_dataset("csv", data_files="/home/jnlp/minhnt/AdvML/custom_dataset/train_labels.csv", split="train")

# dataset = dataset[:100]

# train_dataset = dataset["train"]
# val_dataset = dataset["validation"]

answers = [item for item in dataset["answer"]]
unique_labels = list(set(answers))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}


model_name = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_name)

def preprocess_data(examples):
    pids = examples["file"]

    image_paths = [
        f"/home/jnlp/minhnt/AdvML/custom_dataset/train/{pid}" for pid in pids
    ]

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    texts = examples["question"]

    try:
        encoding = processor(
            images, texts, padding="max_length", truncation=True, return_tensors="pt"
        )
    except Exception as e:
        print(f"Error {e} in processor , will skip this batch")
        return {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "pixel_mask": [],
        }

    for k, v in encoding.items():
        encoding[k] = v.squeeze()

    targets = []

    for answer in examples["answer"]:
        target = torch.zeros(len(id2label))
        answer_id = label2id[answer]
        target[answer_id] = 1.0
        targets.append(target)

    encoding["labels"] = targets
    return encoding

def preprocess_test_data(examples):
    pids = examples["file"]

    image_paths = [
        f"/home/jnlp/minhnt/AdvML/custom_dataset/test/{pid}" for pid in pids
    ]

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    texts = examples["question"]

    try:
        encoding = processor(
            images, texts, padding="max_length", truncation=True, return_tensors="pt"
        )
    except Exception as e:
        print(f"Error {e} in processor , will skip this batch")
        return {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "pixel_mask": [],
        }

    for k, v in encoding.items():
        encoding[k] = v.squeeze()
    # targets = []

    # for answer in examples["answer"]:
    #     target = torch.zeros(len(id2label))
    #     answer_id = label2id[answer]
    #     target[answer_id] = 1.0
    #     targets.append(target)

    # encoding["labels"] = targets
    return encoding
    
def infer():
    ckpt_path = "/home/jnlp/minhnt/AdvML/dl_ckpt/ViLT-ft/checkpoint-423"
    model = ViltForQuestionAnswering.from_pretrained(
        ckpt_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # model.load_state_dict(torch.load(f"{ckpt_path}/pytorch_model.bin"))
    
    test_dataset = load_dataset("csv", data_files="/home/jnlp/minhnt/AdvML/custom_dataset/test.csv", split="train")
    
    args = TrainingArguments(
        output_dir="./tmp_inference",
        per_device_eval_batch_size=16,
    )
    
    processed_test_dataset = test_dataset.map(
        preprocess_test_data,
        batched=True,
        remove_columns=[
            "question",
            "file",
            "id"
        ]
    )
    
    trainer = Trainer(
        model=model,
        args=args,
    )
    
    predictions = trainer.predict(processed_test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
    res = []
    for i, pred in enumerate(preds):
        answer = id2label[pred.item()]
        print(f"Q: {test_dataset[i]['question']}")
        print(f"A: {answer}")
        print("-----")
        res.append({
            "id": int(test_dataset[i]['id']),
            "file": test_dataset[i]['file'],
            "question": test_dataset[i]['question'],
            "answer": answer,
            "explanation": "tmp"
        })
        
    res_df = pd.DataFrame(res)
    res_df.to_csv("/home/jnlp/minhnt/AdvML/results/test/vilt_ft_test_output.csv", index=False)
        
    
def train():
    processed_dataset = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=[
            "question",
            "answer",
            "file",
            "explanation",
            "id"
        ],
    )


    data_collator = DefaultDataCollator()
    model = ViltForQuestionAnswering.from_pretrained(
        model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


    training_args = TrainingArguments(
        output_dir="/home/jnlp/minhnt/AdvML/dl_ckpt/ViLT-ft",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_steps=200,
        logging_steps=200,
        learning_rate=5e-5,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    split_ratio = 0.9
    split_idx = round(len(processed_dataset) * split_ratio)
    train_ds = processed_dataset.select(range(split_idx))
    eval_ds = processed_dataset.select(range(split_idx, len(processed_dataset)))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
    )

    trainer.train()
    
if __name__ == "__main__":
    infer()