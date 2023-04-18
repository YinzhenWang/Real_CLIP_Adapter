import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizerFast, CLIPImageProcessor

from adapter_configs import configs
from data_pre import get_train_dataset
from model import CLIPALL
from utils import create_logger


sys.path.append('../')


def finetune(data_loader, model, epochs, optimizer, criterion, output_dir, device):
    for _ in range(epochs):

        model.train()
        with tqdm(data_loader, desc='train {}'.format(_)) as loop:
            for x in loop:
                for key, value in x.items():
                    x[key] = torch.tensor(np.array(value)).to(device, non_blocking=True)

                inputs = {}
                inputs["pixel_values"] = torch.squeeze(x['pixel_values'])
                inputs["input_ids"] = x['input_ids']
                inputs["attention_mask"] = (1 - x['attention_mask'])
                outputs = model(**inputs)
                vision_cls = outputs.image_embeds
                language_cls = outputs.text_embeds

                loss, logit1, logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
                logger.info('train_epoch{}_adapter_{}_loss_{}'.format(_, text_configname, loss.item()))

        model.clipvision.save_adapter(output_dir + "_v", "LoRA")
        model.cliptext.save_adapter(output_dir + "_t", "mam")
        torch.save(model.state_dict(), output_dir + '_v/ckpt.pth')


if __name__ == "__main__":
    output_dir = "./output_adapter"
    checkpoint_dir = "./lora_ckpt"
    log_path = "log.txt"
    text_configname = "mam"
    epochs = 20

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(os.path.join(output_dir, "vl_finetune_log_" + timestamp + ".txt"),
                           add_stream=False)
    print('Save path:', output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model and Add transform for image input and tokenizer for text input
    backbone = "openai/clip-vit-base-patch16"
    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
    model = CLIPALL(backbone)

    # Configs for Dataloader
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 16
    args['num_workers'] = 8
    data_loader_train = get_train_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                          opt_kwargs)

    # Load adapter in vision and text model
    model.clipvision.load_adapter(checkpoint_dir)
    model.clipvision.set_active_adapters("LoRA")
    model.clipvision.train_adapter("LoRA")

    model.cliptext.add_adapter(text_configname, config=configs[text_configname])
    model.cliptext.train_adapter(text_configname)
    model.cliptext.set_active_adapters(text_configname)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "LoRA" in name or "mam" in name:
            param.requires_grad = True
        print(name, param.requires_grad)

    criterion = model.criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    model.to(device)

    finetune(data_loader_train, model, epochs, optimizer, criterion, output_dir, device)


