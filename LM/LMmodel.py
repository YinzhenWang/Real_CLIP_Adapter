import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel

class CLIPClassification(nn.Module):

    def __init__(self, input_size,dropout):
        super().__init__()
        self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        self.classifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, 2)
        )


    def forward(self, inputs, **kwargs):
        outputs = self.cliptext(**inputs)
        x = outputs.pooler_output
        x = self.classifer(x)
        return x