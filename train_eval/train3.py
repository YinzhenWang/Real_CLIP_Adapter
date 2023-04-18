import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPImageProcessor

from data_pre import get_train_dataset
from model3 import CLIPALL
from utils import create_logger

sys.path.append('../')


if __name__ == "__main__":
    exp = "share_exp3"
    output_dir = "./" + exp
    checkpoint_dir = "./lora_ckpt"

    vision_configname = "LoRA"
    text_configname = "LoRA"

    backbone = "openai/clip-vit-base-patch16"

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(os.path.join(output_dir, "vl3_finetune_log_" + timestamp + ".txt"),
                           add_stream=False)
    print('Save path:', output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the model, transform for image input and tokenizer for text input
    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizer.from_pretrained(backbone)
    model = CLIPALL(backbone)

    # Configs for dataloader
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 16
    args['num_workers'] = 8
    data_loader_train = get_train_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "cliptext" in name and "encoder" not in name:
            param.requires_grad = True
        if "text_projection" in name or "LoRA" in name:
            param.requires_grad = True
        print(name, param.shape, param.requires_grad)

    criterion = model.criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    model.to(device)
    for _ in range(20):

        model.train()
        with tqdm(data_loader_train, desc='train {}'.format(_)) as loop:
            cnt = 0
            for x in loop:
                cnt += 1
                if cnt % 2 == 0.0:
                    logger.info('train_epoch{}_adapter_{}_loss_{}'.format(_, vision_configname + "+" + text_configname, loss.item()))
                    torch.save(model.state_dict(), output_dir + '_t/ckpt.pth')
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
        logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_, vision_configname + "+" + text_configname, loss.item()))
        # model.clipvision.save_adapter(output_dir+"_v", vision_configname)
        # model.cliptext.save_adapter(output_dir+"_t", text_configname+"_t")
        torch.save(model.state_dict(), output_dir + '_t/ckpt.pth')
