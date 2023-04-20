import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
import data_pre
from transformers import CLIPModel, XLMRobertaTokenizer, CLIPTokenizer,CLIPImageProcessor
import logging
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig
from timm.scheduler.cosine_lr import CosineLRScheduler
import os


class ClipLoss(nn.Module):

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss, logits_per_image, logits_per_text


configs = {
        "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
        "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
        "LoRA": LoRAConfig(r=8, alpha=16),
        "IA3": IA3Config(),
        "mam": MAMConfig(),
        "unipelt": UniPELTConfig()
    }


if __name__ == "__main__":
    exp = "finetune"
    output_dir = "./"+exp


    log_path = "log_"+exp+".txt"



    backbone = "openai/clip-vit-base-patch16"
    task_handler = data_pre.RetrievalHandler()



    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    

    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizer.from_pretrained(backbone)
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 16
    args['num_workers'] = 8
    data_loader_train = data_pre.get_train_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    # data_loader_val = data_pre.get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)


    config_name = "LoRA"
    config = configs[config_name]

    model = CLIPModel.from_pretrained(backbone)



    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'logit_scale' in name:
            param.requires_grad = False
        print(name, param.requires_grad)


    criterion = ClipLoss()


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))
    epochs = 50
    warmup_epochs = 1
    n_iter_per_epoch = len(data_loader_train)
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


    model.to(device)
    for _ in range(epochs):

        model.train()
        with tqdm(data_loader_train, desc='train {}'.format(_)) as loop:
            for x in loop:
                for key,value in x.items():
                    x[key] = torch.tensor(np.array(value)).to(device, non_blocking=True)

                inputs = {}
                inputs["pixel_values"] =  torch.squeeze(x['pixel_values'])
                inputs["input_ids"] = x['input_ids']
                inputs["attention_mask"] =(1-x['attention_mask'])
                outputs = model(**inputs)
                vision_cls = outputs.image_embeds
                language_cls = outputs.text_embeds

                loss,logit1,logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss)
        logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_,'no',loss))
        ext_test_stats, task_key = data_pre.evaluate(data_loader_test, model, device, task_handler)
        print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
        mim_checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': _,
                          'config': config}
        output_path = os.path.join(output_dir, f'ckpt_epoch_{_}')
        os.makedirs(output_path, exist_ok=True)
        mim_checkpoint_path = os.path.join(output_path, f'mim_epoch_{_}.pth')
        torch.save(mim_checkpoint, mim_checkpoint_path)