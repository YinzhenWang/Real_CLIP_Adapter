import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CLIPVisionClassification(nn.Module):

    def __init__(self, input_size, class_num, dropout):
        super().__init__()
        self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.classifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, class_num)
        )

    def forward(self, inputs, **kwargs):
        outputs = self.clipvision(**inputs)
        x = outputs.pooler_output
        x = self.classifer(x)
        return x
