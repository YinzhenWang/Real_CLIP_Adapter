import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizerFast, CLIPImageProcessor


import data_pre
from model_itm2 import CLIPAll_SimpleCrossAttn
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig


import logging

def create_logger(filename, add_stream=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

configs = {
    "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
    "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
    "LoRA": LoRAConfig(r=8, alpha=16),
    "IA3": IA3Config(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig()
}







def finetune(data_train_loader, data_test_loader, model, epochs, optimizer, criterion, output_dir, device):
    for _ in range(epochs):

        model.train()
        with tqdm(data_train_loader, desc='train {}'.format(_)) as loop:
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
                cliploss = outputs.cliploss
                itmloss = outputs.itmloss
                loss = outputs.loss

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()

                loop.set_postfix({'cliploss': cliploss, 'itmloss': itmloss})
        logger.info('train_epoch{}_loss_{}'.format(_, loss.item()))

        ext_test_stats, task_key = data_pre.evaluate(data_test_loader, model, device, data_pre.RetrievalHandler())
        torch.save(model.state_dict(), output_dir + '/ckpt.pth')


if __name__ == "__main__":
    exp = "itm_0.1"
    output_dir = "./"+exp
    text_checkpoint_dir = "../mlm"
    vision_checkpoint_dir = "../mim_100"
    vision_configname = "LoRA"
    text_configname = "mam"
    epochs = 50



    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(os.path.join(output_dir, "_log_" + timestamp + ".txt"),
                           add_stream=False)
    print('Save path:', output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model and Add transform for image input and tokenizer for text input
    backbone = "openai/clip-vit-base-patch16"
    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
    model = CLIPAll_SimpleCrossAttn(backbone)

    # Configs for Dataloader
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 64
    args['num_workers'] = 8
    data_loader_train = data_pre.get_train_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                          opt_kwargs)
    data_loader_test = data_pre.get_test_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                          opt_kwargs)

    # Load adapter in vision and text model
    model.clipvision.load_adapter(vision_checkpoint_dir)
    model.clipvision.set_active_adapters("LoRA")

    model.cliptext.load_adapter(text_checkpoint_dir)
    model.cliptext.set_active_adapters("mam")
    

    criterion = model.criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    model.to(device)

    for name, param in model.named_parameters():
      if "cliptext" in name or "clipvision" in name:
        param.requires_grad = False
      print(name, param.requires_grad)

    finetune(data_loader_train, data_loader_test, model, epochs, optimizer, criterion, output_dir, device)

