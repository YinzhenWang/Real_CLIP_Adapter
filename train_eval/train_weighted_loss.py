import os
import sys
import time

import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import AverageMeter
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from transformers import CLIPImageProcessor,CLIPTokenizer

from Visionmodel import CLIPWeightedLOSS, CLIPVisionMasked
from mask_generator import MaskGenerator
import logging
import data_pre
import numpy as np
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig

weight = 0.1



configs = {
        "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
        "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
        "LoRA": LoRAConfig(r=8, alpha=16),
        "IA3": IA3Config(),
        "mam": MAMConfig(),
        "unipelt": UniPELTConfig()
    }

def create_logger(filename, add_stream=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

args = {}
args['batch_size'] = 64
args['num_workers'] = 8
opt_kwargs = {}
output_dir = './masked_image_ckpt'


sys.path.append('../')  # could be comment out


def masked_modeling(data_path, configname, config, epochs, warmup_epochs, mask_patch_size, model_patch_size,
                    mask_ratio, output_dir, batch_size=16, check_grad=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mask_generator = MaskGenerator(input_size=224, mask_patch_size=mask_patch_size,
                                   model_patch_size=model_patch_size, mask_ratio=mask_ratio)

    # Train set: 34745; Val set: 3923
    print("Loading dataset...")
    # trainset = get_imagenet_mini(data_path, 'train', transform, mask_generator, 
    #                              batch_size=batch_size, num_workers=32, max_len=35000)
    # valset = get_imagenet_mini(data_path, 'val', transform, mask_generator, 
    #                              batch_size=batch_size, num_workers=32, max_len=4000)
    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    trainset = data_pre.get_train_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    valset = data_pre.get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    testset = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)

    # Define model
    vision_encoder = CLIPVisionMasked(dropout_rate=0.1)
    model = CLIPWeightedLOSS(vision_encoder, 16)

    vision_encoder.clipvision.add_adapter(configname, config=config)
    vision_encoder.clipvision.train_adapter(configname)
    vision_encoder.clipvision.set_active_adapters(configname)
    if check_grad:
        print("================== Gradient Info ==================")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    model.to(device)
    logger.info(model)

    # Wrap model in DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        logger.info('Multi-GPU training.')
        print('Start multi-GPU training.')
        model = DataParallel(model)
    else:
        logger.info('Single GPU training.')
        print('Start single GPU training.')

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))

    # Calculate parameter
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
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'LoRA' in name:
            param.requires_grad = True
        print(name, param.requires_grad)

    logger.info("Start training")

    for epoch in range(1, epochs + 1):
        loss_meter = AverageMeter()

        # Train model
        optimizer.zero_grad()
        model.train()
        with tqdm(trainset, desc='train_epoch{}_adapter_{}'.format(epoch, configname)) as loop:
            
            for idx, x in enumerate(loop):
                img = torch.tensor(np.array(x['pixel_values'])).to(device)
                img = torch.squeeze(img)
                
                input_ids = torch.tensor(np.array(x['input_ids'])).to(device)
                attention_mask = torch.tensor(np.array(x['attention_mask'])).to(device)

                inputs = {}
                inputs["pixel_values"] =  img
                inputs["input_ids"] = input_ids
                inputs["attention_mask"] = attention_mask
                
                output = model(**inputs)
                img_rec = output.x_rec

                cliploss = output.loss
                mask = output.mask.repeat_interleave(model_patch_size, 1).repeat_interleave(model_patch_size, 2).unsqueeze(1).contiguous()

                loss_recon = F.l1_loss(img, img_rec, reduction='none')
                loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3

                loss = cliploss*weight + loss_recon*(1-weight)

                if loss != 0:
                    loss.backward()

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
                loss_meter.update(loss.item(), img.size(0))
                
                loop.set_postfix({'recloss':loss_recon,'cliploss':cliploss})
        ext_test_stats, task_key = data_pre.evaluate(testset, model, device, data_pre.RetrievalHandler())
        logger.info('train_epoch{}_adapter_{}, loss:{}'.format(epoch, configname, loss_meter.avg))

        # Validate model
        # if epoch % 5 == 0:
        val_loss_meter = AverageMeter()
        model.eval()
        with torch.no_grad():
            with tqdm(valset, desc='val_epoch{}_adapter_{}'.format(epoch, configname)) as loop:
                for idx, x in enumerate(loop):

                    img = torch.tensor(np.array(x['pixel_values'])).to(device)
                    img = torch.squeeze(img)
                    
                    input_ids = torch.tensor(np.array(x['input_ids'])).to(device)
                    attention_mask = torch.tensor(np.array(x['attention_mask'])).to(device)

                    inputs = {}
                    inputs["pixel_values"] =  img
                    inputs["input_ids"] = input_ids
                    inputs["attention_mask"] = attention_mask
                    
                    output = model(**inputs)
                    img_rec = output.x_rec

                    mask = output.mask.repeat_interleave(model_patch_size, 1).repeat_interleave(model_patch_size, 2).unsqueeze(1).contiguous()

                    loss_recon = F.l1_loss(img, img_rec, reduction='none')
                    loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3

                    loss = cliploss*weight + loss_recon*(1-weight)

                    val_loss_meter.update(loss.item(), img.size(0))
                    loop.set_postfix({'recloss':loss_recon,'cliploss':cliploss})
            logger.info('val_epoch{}_adapter_{}, loss:{}'.format(epoch, configname, val_loss_meter.avg))

        # Save checkpoint
        
        # Unwrap model from DataParallel
        if isinstance(model, DataParallel):
            saved_model = model.module
        else:
            saved_model = model
        # Save adapter and mim
        output_path = os.path.join(output_dir, f'ckpt_epoch_{epoch}')
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving checkpoint at epoch {epoch} to {output_path}")
        vision_encoder.clipvision.save_pretrained(output_path)
        vision_encoder.clipvision.save_adapter(output_path, f"{configname}")

        mim_checkpoint = {'model': saved_model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'config': config}
        mim_checkpoint_path = os.path.join(output_path, f'mim_epoch_{epoch}.pth')
        torch.save(mim_checkpoint, mim_checkpoint_path)


if __name__ == "__main__":
    # adapter config
    config_name = "LoRA"
    config = configs[config_name]

    # logger and model saved dir
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    output_dir = os.path.join('./Log', f'{config_name}_mim_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(os.path.join(output_dir, "masked_modeling_log_" + timestamp + ".txt"),
                           add_stream=False)
    print('Save path:', output_dir)

    train_epochs = 50
    warmup_epochs = 1
    data_path = '../dataset/data/imagenet'
    mask_patch_size = 32
    model_patch_size = 16
    mask_ratio = 0.6
    batch_size = 64

    masked_modeling(data_path, config_name, config, train_epochs, warmup_epochs,
                    mask_patch_size, model_patch_size, mask_ratio, output_dir, batch_size, check_grad=False)
