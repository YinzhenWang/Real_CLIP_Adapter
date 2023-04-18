import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig, LoRAConfig
from datasets import load_dataset
from LMmodel import VitForText

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
device = "cuda:5"

train_dataset = load_dataset("glue", "sst2", split="train")
test_dataset = load_dataset("glue", "sst2", split="test")

train_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

model = VitForText().to(device)
# config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
# config = LoRAConfig(r=8, alpha=16)

# model.cliptext.add_adapter("LoRA", config=config)
# model.cliptext.set_active_adapters("LoRA")
# model.cliptext.train_adapter("LoRA")

for name, param in model.named_parameters():
    print(name, param.requires_grad)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

for _ in range(20):
    cnt = 0
    total = 0

    model.train()
    with tqdm(train_dataloader, desc='train {}'.format(_)) as loop:
        for x in loop:
            sentences = x['sentence']
            labels = x['label'].to(device)
            # print(sentences)
            # print(labels)
            inputs = tokenizer(list(sentences), padding=True, return_tensors="pt").to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            total += len(labels)
            classification = (out[..., 0] < out[..., 1]).long().flatten()
            cnt += (classification == labels).int().sum().item()
            optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss,acc = cnt/total)

    model.eval()
    with tqdm(test_dataloader, desc='test {}'.format(_)) as loop:
        for x in loop:
            sentences = x['sentence']
            labels = x['label'].to(device)
            # print(sentences)
            # print(labels)
            inputs = tokenizer(list(sentences), padding=True, return_tensors="pt").to(device)
            # print(inputs)
            out = model(inputs)

            total += len(labels)
            classification = (out[..., 0] < out[..., 1]).long().flatten()
            cnt += (classification == labels).int().sum().item()

            loop.set_postfix(acc = cnt/total)

torch.save(model.cliptext.state_dict(), 'cliptext.pth')