import torch
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, texts, hazard_labels, product_labels, tokenizer, max_len=128):
        self.texts = texts.values
        self.hazard_labels = hazard_labels.values
        self.product_labels = product_labels.values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'hazard_label': torch.tensor(self.hazard_labels[idx], dtype=torch.long),
            'product_label': torch.tensor(self.product_labels[idx], dtype=torch.long)
        }