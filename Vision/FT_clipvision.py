import os
import time

import torch
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig
from timm.utils import AverageMeter

from Visionmodel import CLIPVisionClassification
from utils import create_logger


def train(train_set, test_set, processor, model, configname, batch_size, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(epochs):
        # Train
        total, cnt, bz = 0, 0, 0
        train_loss = AverageMeter()
        tmp_img, tmp_label = [], []
        model.train()
        with tqdm(train_set, desc='train_epoch{}_adapter_{}'.format(epoch, configname)) as loop:
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

                    train_loss.update(loss.item(), len(tmp_label))
                    loop.set_postfix(loss=train_loss.avg, acc=cnt / total)

                    bz = 0
                    tmp_img = []
                    tmp_label = []
        logger.info('train_epoch{}_adapter_{}, acc:{}'.format(epoch, configname, cnt / total))

        # Evaluate
        total, cnt, bz = 0, 0, 0
        tmp_img = []
        tmp_label = []
        model.eval()
        with tqdm(test_set, desc='test_epoch{}_adapter_{}'.format(epoch, configname)) as loop:
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
        logger.info('test_epoch{}_adapter_{}, acc:{}'.format(epoch, configname, cnt / total))

    model.clipvision.save_adapter("./{}".format(configname), "{}".format(configname))


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

    trainset = CIFAR100(root=os.path.expanduser("~/.cache"), train=True, download=True)
    testset = CIFAR100(root=os.path.expanduser("~/.cache"), train=False, download=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"  #
    train_epochs = 20

    configs = {
        "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
        "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
        "LoRA": LoRAConfig(r=8, alpha=16),
        "IA3": IA3Config(),
        "mam": MAMConfig(),
        "unipelt": UniPELTConfig()
    }

    for configname, config in configs.items():
        # logger and model saved dir
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        output_dir = os.path.join('./Log', f'{configname}_finetune_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        logger = create_logger(os.path.join(output_dir, "finetune_clipvision_log_" + timestamp + ".txt"),
                               add_stream=False)
        print('Save path:', output_dir)

        model = CLIPVisionClassification(768, len(trainset.classes), 0.2)

        model.clipvision.add_adapter(configname, config=config)
        model.clipvision.train_adapter(configname)
        model.clipvision.set_active_adapters(configname)
        model.to(device)

        train(trainset, testset, processor, model, configname, 32, train_epochs)
