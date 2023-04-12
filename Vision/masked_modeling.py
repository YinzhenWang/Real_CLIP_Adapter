import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100, ImageNet

from transformers import AutoProcessor
from transformers.adapters import LoRAConfig

from Vision.mask_generator import MaskGenerator
from Visionmodel import MIM, CLIPVisionMasked
from utils import create_logger

from PIL import Image


def masked_modeling(configname, config, epochs, check_grad=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # trainset = CIFAR100(root=os.path.expanduser("~/.cache"), train=True, download=True)
    # testset = CIFAR100(root=os.path.expanduser("~/.cache"), train=False, download=True)
    # trainset = ImageNet(root="dataset/data/imagenet", split="train")

    vision_encoder = CLIPVisionMasked(dropout_rate=0.2)
    model = MIM(vision_encoder, 16)

    vision_encoder.clipvision.add_adapter(configname, config=config)
    vision_encoder.clipvision.train_adapter(configname)
    vision_encoder.clipvision.set_active_adapters(configname)
    if check_grad:
        print("================== Gradient Info ==================")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    model.to(device)
    logger.info(model)

    # TODO: add lr scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.05,
                                  betas=(0.9, 0.999))

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {n_params}, trained params: {n_train_params}")

    logger.info("Start training")

    batch_size = 2
    model_patch_size = 16
    mask_generator = MaskGenerator(input_size=224, model_patch_size=model_patch_size)
    test_mask = mask_generator()

    for _ in range(epochs):
        bz = 0
        tmp_img = []
        tmp_mask = []
        optimizer.zero_grad()
        model.train()
        # with tqdm(trainset, desc='train_epoch{}_adapter_{}'.format(_, configname)) as loop:
        #     for image, class_id in loop:
        image = Image.open('../dataset/data/imagenet/train/n01443537/n01443537_130.JPEG')
        image = np.array(image)
        if True:
                # if bz <= batch_size:
                #     bz += 1
                    tmp_img.append(image)
                    random_mask = mask_generator()
                    tmp_mask.append(random_mask)
                # if bz > batch_size:
                    inputs = processor(images=tmp_img, return_tensors="pt", padding=True).to(device)
                    mask = torch.tensor(tmp_mask).to(device)
                    x = inputs['pixel_values']
                    x_rec = model(x, mask)

                    mask = mask.repeat_interleave(model_patch_size, 1).repeat_interleave(model_patch_size, 2).unsqueeze(
                        1).contiguous()

                    loss_recon = F.l1_loss(x, x_rec, reduction='none')
                    loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3

                    if loss != 0:
                        loss.backward()
                    optimizer.step()

                    bz = 0
                    tmp_img = []
                    tmp_mask = []

                    loop.set_postfix(loss=loss.item())
        # logger.info('train_epoch{}_adapter_{}, loss:{}'.format(_, configname, loss))


if __name__ == "__main__":
    logger = create_logger("masked_modeling_log.txt", add_stream=False)

    configs = {"LoRA": LoRAConfig(r=8, alpha=16)}
    config_name = "LoRA"

    train_epochs = 5
    masked_modeling(config_name, configs[config_name], train_epochs, check_grad=False)
