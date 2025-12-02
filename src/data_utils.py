import yaml

class PromptTemplate():
    def __init__(self, template_path):
        with open(template_path, 'r') as file:
            self.templates = yaml.safe_load(file)

    def get_templates(self):
        return self.templates

class MyDataLoader():
    def __init__(self, image_path, data_path, split='train'):
        self.image_path = image_path
        self.data_path = data_path
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        # Implement data loading logic here
        if self.split == 'train_split':
            # path = f"{self.data_path}/train_split.csv"
            image_dir = f"{self.image_path}/train/"
        elif self.split == 'train_all':
            # path = f"{self.data_path}/train_labels.csv"
            image_dir = f"{self.image_path}/train/"
        elif self.split == 'dev':
            # path = f"{self.data_path}/dev_split.csv"
            image_dir = f"{self.image_path}/train/"
        elif self.split == 'test':
            # path = f"{self.data_path}/test_non_labels.csv"
            image_dir = f"{self.image_path}/test/"
        elif self.split == 'dev_obj_detect':
            # path = f"{self.image_path}/dev_split.csv"
            image_dir = f"{self.image_path}/dev_obj_detect/"
        elif self.split == 'dev_region_cap':
            # path = f"{self.data_path}/dev_split.csv"
            image_dir = f"{self.image_path}/dev_region_cap/"
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        # Load data from CSV
        import pandas as pd
        data = pd.read_csv(self.data_path)
        return data, image_dir

    def get_batch(self, batch_size):
        # Implement batching logic here
        pass