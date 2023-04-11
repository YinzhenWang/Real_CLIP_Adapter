import os
from tqdm import tqdm
import torch
import logging

from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
from torchvision.datasets import Caltech101, CIFAR10, CIFAR100
from Visionmodel import CLIPVisionClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

trainset = CIFAR100(root=os.path.expanduser("~/.cache"), train=True, download=True)
testset = CIFAR100(root=os.path.expanduser("~/.cache"), train=False, download=True)

model = CLIPVisionClassification(768, len(trainset.classes), 0.2)
config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
model.clipvision.add_adapter("adapter", config=config)
model.clipvision.train_adapter("adapter")
model.clipvision.set_active_adapters("adapter")
model.to(device)

print("================== Gradient Info ==================")
for name, param in model.named_parameters():
    print(name, param.requires_grad)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

print("\nAll Classes:", trainset.classes)
batch_size = 16
train_epoch = 5


def finetune(trainset, testset, model, train_epoch):
    for _ in range(train_epoch):
        total = 0
        cnt = 0
        bz = 0
        tmp_img = []
        tmp_label = []
        with tqdm(trainset, desc='train {}'.format(_)) as loop:
            for image, class_id in loop:
                bz += 1
                if bz <= batch_size:
                    tmp_img.append(image)
                    tmp_label.append(class_id)
                if bz > batch_size:
                    inputs = processor(images=tmp_img, return_tensors="pt", padding=True).to(device)
                    outputs = model(inputs)
                    label = torch.tensor(tmp_label).to(device)

                    loss = criterion(outputs, label)
                    total += len(label)
                    cnt += (torch.argmax(outputs, dim=1) == label).int().sum().item()

                    optimizer.zero_grad()
                    if loss != 0:
                        loss.backward()
                    optimizer.step()
                    bz = 0
                    tmp_img = []
                    tmp_label = []
                    loop.set_postfix(loss=loss.item(), acc=cnt / total)
        total = 0
        cnt = 0
        bz = 0
        tmp_img = []
        tmp_label = []
        with tqdm(testset, desc='test {}'.format(_)) as loop:
            for image, class_id in loop:
                bz += 1
                if bz <= batch_size:
                    tmp_img.append(image)
                    tmp_label.append(class_id)
                if bz > batch_size:
                    inputs = processor(images=tmp_img, return_tensors="pt", padding=True).to(device)
                    outputs = model(inputs)
                    label = torch.tensor(tmp_label).to(device)

                    total += len(label)
                    cnt += (torch.argmax(outputs, dim=1) == label).int().sum().item()

                    bz = 0
                    tmp_img = []
                    tmp_label = []
                    loop.set_postfix(acc=cnt / total)

    model.clipvision.save_adapter("./Visionadapter", "Visionadapter")


if __name__ == '__main__':
    finetune(trainset, testset, model, train_epoch)
