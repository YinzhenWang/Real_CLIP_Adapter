from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
from torchvision.datasets import Caltech101, CIFAR10, CIFAR100
import os
from tqdm import tqdm
import torch

dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
config = MAMConfig()
# config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
device = "cuda:5"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True, return_dict=True)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

'''
model.text_model.load_adapter("./LM/toyadapter")
model.text_model.train_adapter("mam_adapter")
model.text_model.set_active_adapters("mam_adapter")
'''


model.to(device)
total = 0
top1_cnt = 0
top3_cnt = 0
top10_cnt = 0
with tqdm(dataset, desc='train') as loop:
    for image, class_id in loop:

        inputs = processor(
            text=[f"a photo of a {c}" for c in dataset.classes],
            images=image, return_tensors="pt", padding=True
        ).to(device)

        # Calculate features
        with torch.no_grad():
            outputs = model(**inputs)
            print(outputs.vision_model_output.pooler_output.shape)
            break
