import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForSequenceClassification

class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, x):
        return self.layers(x)

class CustomModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.classifier = CustomClassificationHead(hidden_size, config.num_labels)

    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        pooled_output = outputs.pooler_output  # [CLS] токен
        logits = self.classifier(pooled_output)
        return {'logits': logits}
