import copy
import os
import sys
import numpy as np
from PIL import Image

import torch
from transformers import CLIPImageProcessor

from Visionmodel import CLIPVisionMasked, MIM
from data_imagenet_mini import get_imagenet_mini
from mask_generator import MaskGenerator
from adapter_configs import configs

sys.path.append('../')


def viz_reconstruction(data_path, output_dir, checkpoint_path, configname, config, model_patch_size=16,
                       mask_patch_size=16, mask_ratio=0.6):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare validation data
    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mask_generator = MaskGenerator(input_size=224, mask_patch_size=mask_patch_size,
                                   model_patch_size=model_patch_size, mask_ratio=mask_ratio)

    valset = get_imagenet_mini(data_path, 'val', transform, mask_generator,
                               batch_size=2, num_workers=8, max_len=6)

    # Define model and load the checkpoint
    vision_encoder = CLIPVisionMasked(dropout_rate=0.1)
    model = MIM(vision_encoder, 16)

    vision_encoder.clipvision.load_adapter(output_dir, config=config)
    vision_encoder.clipvision.set_active_adapters(configname)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Visualize the first image and its reconstruction in each batch
    viz_imgs = []
    for img, mask in valset:
        img, mask = img.to(device), mask.to(device)
        img_rec = model(img, mask)

        # Apply mask to image
        show_img = img[0].detach().cpu().numpy()
        show_img = (show_img * 255).astype(np.uint8).transpose(1, 2, 0)
        show_mask = mask[0].detach().cpu().numpy()
        show_mask = np.repeat(np.repeat(show_mask, 16, axis=1), 16, axis=0)
        show_img_masked = copy.deepcopy(show_img)
        show_img_masked[show_mask == 1] = [0, 0, 0]

        show_img_rec = img_rec[0].detach().cpu().numpy()
        show_img_rec = (show_img_rec * 255).astype(np.uint8).transpose(1, 2, 0)
        show_img_rec[show_mask == 0] = [0, 0, 0]
        show_img_rec = show_img_rec + show_img_masked

        show_img_masked[show_mask == 1] = [255, 255, 255]

        viz_imgs.append(np.concatenate([show_img, show_img_masked, show_img_rec], axis=1))

    # Concatenate all images and show them using PIL
    img_array = np.concatenate(viz_imgs, axis=0)
    img = Image.fromarray(img_array)
    img.save('reconstruct_49.jpg')


if __name__ == '__main__':
    data_path = '../dataset/data/imagenet'

    config_name = 'LoRA'
    config = configs[config_name]

    output_dir = "./ckpt_epoch_49"
    checkpoint_path = os.path.join(output_dir, 'mim_epoch_49.pth')

    viz_reconstruction(data_path, output_dir, checkpoint_path, config_name, config)
