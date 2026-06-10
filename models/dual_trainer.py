# models/dual_trainer.py
import torch
from torch import nn
from transformers import Trainer

class DualModelTrainer(Trainer):
    def __init__(self, target_column, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop(self.target_column)
        other_column = 'product_label' if self.target_column == 'hazard_label' else 'hazard_label'
        if other_column in inputs:
            inputs.pop(other_column)
        outputs = model(**inputs)
        dynamic_weights = self.class_weights.to(outputs.logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=dynamic_weights)
        loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss