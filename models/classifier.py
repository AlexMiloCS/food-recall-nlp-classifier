import torch.nn as nn
from transformers import AutoModel

class FoodHazardClassifier(nn.Module):
    def __init__(self, model_name, num_hazard_classes, num_product_classes, hf_token=None, dropout_rate=0.3):
        super(FoodHazardClassifier, self).__init__()
        
        # 1. Φόρτωση του Base Model
        self.encoder = AutoModel.from_pretrained(model_name, token=hf_token)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. Βελτιωμένο Hazard Head (Multi-Layer Perceptron)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),  # Μη γραμμικότητα
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_hazard_classes)
        )
        
        # 4. Βελτιωμένο Product Head (Multi-Layer Perceptron)
        self.product_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),  # Μη γραμμικότητα
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_product_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Εξαγωγή του πρώτου token ([CLS] για BERT, <s> για RoBERTa)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Περνάμε το CLS token από το αρχικό Dropout
        cls_output = self.dropout(cls_output)
        
        # Εξαγωγή προβλέψεων μέσα από τα βελτιωμένα heads
        hazard_logits = self.hazard_head(cls_output)
        product_logits = self.product_head(cls_output)
        
        return hazard_logits, product_logits