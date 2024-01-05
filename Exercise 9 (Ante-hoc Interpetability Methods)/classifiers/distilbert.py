from pathlib import Path
from transformers import DistilBertForTokenClassification, DistilBertForSequenceClassification
import sys
import os  # noqa
# os.chdir('..')
sys.path.insert(0, "")  # noqa
import torch
torch.manual_seed(0)


SELECTOR_CHECKPOINT = 'sarah-krebs/iML-distilbert-base-uncased-select'
PREDICTOR_CHECKPOINT = 'sarah-krebs/iML-distilbert-base-uncased-predict'


class Selector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForTokenClassification.from_pretrained(SELECTOR_CHECKPOINT)
        for name, param in self.model.named_parameters():
            # freeze everything except classifier
            if 'classifier' not in name and '5' not in name: param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits.permute(0, 2, 1)


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(PREDICTOR_CHECKPOINT)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
