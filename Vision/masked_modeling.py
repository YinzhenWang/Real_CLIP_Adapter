import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import CLIPImageProcessor
from transformers.adapters import LoRAConfig

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import AverageMeter

from Vision.data_imagenet_mini import get_imagenet_mini
from Vision.mask_generator import MaskGenerator
from Visionmodel import MIM, CLIPVisionMasked
from utils import create_logger


def masked_modeling(data_path, configname, config, epochs, warmup_epochs, mask_patch_size, model_patch_size,
                    mask_ratio, check_grad=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mask_generator = MaskGenerator(input_size=224, mask_patch_size=mask_patch_size,
                                   model_patch_size=model_patch_size, mask_ratio=mask_ratio)
    trainset = get_imagenet_mini(data_path, 'train', transform, mask_generator, batch_size=16, num_workers=8, max_len=10000)

    vision_encoder = CLIPVisionMasked(dropout_rate=0.1)
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

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.05,
                                  betas=(0.9, 0.999))

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {n_params}, trained params: {n_train_params}")

    # Build lr scheduler
    n_iter_per_epoch = len(trainset)
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=5e-6,
        warmup_lr_init=5e-7,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    logger.info("Start training")

    for epoch in range(epochs):
        loss_meter = AverageMeter()

        optimizer.zero_grad()
        model.train()
        with tqdm(trainset, desc='train_epoch{}_adapter_{}'.format(epoch, configname)) as loop:
            for idx, (img, mask) in enumerate(loop):
                img = img.to(device)
                mask = mask.to(device)
                img_rec = model(img, mask)

                mask = mask.repeat_interleave(model_patch_size, 1).repeat_interleave(model_patch_size, 2).unsqueeze(1).contiguous()

                loss_recon = F.l1_loss(img, img_rec, reduction='none')
                loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3

                if loss != 0:
                    loss.backward()

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
                loss_meter.update(loss.item(), img.size(0))

                loop.set_postfix(loss=loss_meter.avg)

        logger.info('train_epoch{}_adapter_{}, loss:{.4f}'.format(epoch, configname, loss_meter.avg))


if __name__ == "__main__":
    logger = create_logger("masked_modeling_log.txt", add_stream=False)

    configs = {"LoRA": LoRAConfig(r=8, alpha=16)}
    config_name = "LoRA"

    train_epochs = 20
    warmup_epochs = 5
    data_path = '../dataset/data/imagenet'
    mask_patch_size = 16
    model_patch_size = 16
    mask_ratio = 0.4
    masked_modeling(data_path, config_name, configs[config_name], train_epochs, warmup_epochs,
                    mask_patch_size, model_patch_size, mask_ratio, check_grad=False)
