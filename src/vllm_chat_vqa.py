from vllm import LLM
from vllm.assets.image import ImageAsset
from PIL import Image

llm = LLM(model="Qwen/Qwen3-VL-4B-Instruct")
image_pil = ImageAsset('cherry_blossom').pil_image

image_path="/home/jnlp/minhnt/AdvML/custom_dataset"
data_path="/home/jnlp/minhnt/AdvML/custom_dataset/test_similar_dev_question.csv"
split="test"
prompt_path="prompts/fewshot.yaml"

data_loader = MyDataLoader(image_path=image_path, data_path=data_path, split=split)
data, image_dir = data_loader.data
dev_image_dir = "/home/jnlp/minhnt/AdvML/custom_dataset/dev_split_img"


prompt_template = PromptTemplate(template_path=prompt_path)
prompt = prompt_template.get_templates()

for _, sample in data.iterrows():
    id = sample['id']
    dev_img_name = sample['dev_file']
    dev_question = sample['dev_question']
    dev_explanation = sample['dev_explanation']
    dev_answer = sample['dev_answer']
    
    test_img_name = sample['test_file']
    test_question = sample['test_question']
    
    dev_image_pil = Image.open(f"{image_dir}/{dev_img_name}").convert("RGB")
    test_image_pil = Image.open(f"{dev_image_dir}/{test_img_name}").convert("RGB")
    conversation = [
        {"role": "system", "content": prompt['system']},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_pil",
                    "image_pil": dev_image_pil,
                },
                {
                    "type": "text",
                    "text": prompt['user'].format(question=dev_question),
                },
            ],
        },
        {"role": "assistant", "content": prompt['assistant'].format(explanation=dev_explanation, answer=dev_answer)},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_pil",
                    "image_pil": test_image_pil,
                },
                {
                    "type": "text",
                    "text": prompt['user'].format(question=test_question),
                },
            ],
        },
    ]

    # Perform inference and log output.
    outputs = llm.chat(conversation)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        
    break