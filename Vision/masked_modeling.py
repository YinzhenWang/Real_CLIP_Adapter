import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import CLIPImageProcessor
from transformers.adapters import LoRAConfig

from Vision.data_imagenet_mini import get_imagenet_mini
from Vision.mask_generator import MaskGenerator
from Visionmodel import MIM, CLIPVisionMasked
from utils import create_logger


def masked_modeling(configname, config, epochs, check_grad=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_patch_size = 16
    data_path = '../dataset/data/imagenet'
    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mask_generator = MaskGenerator(input_size=224, model_patch_size=model_patch_size)
    trainset = get_imagenet_mini(data_path, 'train', transform, mask_generator, 16, 8, max_len=10000)

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

    for _ in range(epochs):
        total = 0
        total_loss = 0

        optimizer.zero_grad()
        model.train()
        with tqdm(trainset, desc='train_epoch{}_adapter_{}'.format(_, configname)) as loop:
            for img, mask in loop:
                img = img.to(device)
                mask = mask.to(device)
                img_rec = model(img, mask)

                mask = mask.repeat_interleave(model_patch_size, 1).repeat_interleave(model_patch_size, 2).unsqueeze(1).contiguous()

                loss_recon = F.l1_loss(img, img_rec, reduction='none')
                loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3

                if loss != 0:
                    loss.backward()
                optimizer.step()

                batch_size = img.size(0)
                total += batch_size
                total_loss += loss.item() * batch_size

                loop.set_postfix(loss=total_loss / total)
        logger.info('train_epoch{}_adapter_{}, loss:{}'.format(_, configname, total_loss / total))


if __name__ == "__main__":
    logger = create_logger("masked_modeling_log.txt", add_stream=False)

    configs = {"LoRA": LoRAConfig(r=8, alpha=16)}
    config_name = "LoRA"

    train_epochs = 5
    masked_modeling(config_name, configs[config_name], train_epochs, check_grad=False)
