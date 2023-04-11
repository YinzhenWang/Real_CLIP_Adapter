from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig
from torchvision.datasets import Caltech101, CIFAR10, CIFAR100
import os
from tqdm import tqdm
import torch
from Visionmodel import CLIPVisionClassification
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log.txt')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = "cuda:6"

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

configs = {
    "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
    "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
    "LoRA": LoRAConfig(r=8, alpha=16),
    "IA3": IA3Config(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig()
}

for configname,config in configs.items():
    trainset = CIFAR100(root=os.path.expanduser("~/.cache"), train=True, download=True)
    testset = CIFAR100(root=os.path.expanduser("~/.cache"), train=False, download=True)

    model = CLIPVisionClassification(768, len(trainset.classes), 0.2)

    model.clipvision.add_adapter(configname, config=config)
    model.clipvision.train_adapter(configname)
    model.clipvision.set_active_adapters(configname)
    model.to(device)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    batch_size = 32

    for _ in range(10):
        total = 0
        cnt = 0
        bz = 0
        tmp_img = []
        tmp_label = []
        with tqdm(trainset, desc='train_epoch{}_adapter_{}'.format(_,configname)) as loop:
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
                    loop.set_postfix(loss=loss, acc=cnt / total)
        logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_,configname,cnt / total))

        total = 0
        cnt = 0
        bz = 0
        tmp_img = []
        tmp_label = []
        with tqdm(testset, desc='test_epoch{}_adapter_{}'.format(_,configname)) as loop:
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
                    loop.set_postfix(loss=loss, acc=cnt / total)
        logger.info('test_epoch{}_adapter_{}_acc_{}'.format(_, configname, cnt / total))

    model.clipvision.save_adapter("./{}".format(configname), "{}".format(configname))
