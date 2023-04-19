
import math
import os
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_pre
from transformers import CLIPModel,  CLIPTokenizer,CLIPImageProcessor
from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
from model_adapter_stack import CLIPALL


backbones = ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32","openai/clip-vit-large-patch14","openai/clip-vit-large-patch14-336"]


if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
  task_handler = data_pre.RetrievalHandler()

  backbone = "openai/clip-vit-base-patch16"
  
  filedir = "./exp_stack"
  
  vision_configname = "LoRA"
  text_configname = "LoRA"



  model = CLIPALL(backbone)

  

  model.clipvision.load_adapter(filedir+"_v")
  model.clipvision.set_active_adapters(vision_configname)
  model.clipvision.load_adapter(filedir+"_v2")
  model.clipvision.set_active_adapters(vision_configname+"_v2")
  model.cliptext.load_adapter(filedir+"_t")
  model.cliptext.set_active_adapters(text_configname+"_t")

  for name, param in model.named_parameters():
    print(name, param.requires_grad)

  model.load_state_dict(torch.load(filedir+'_v/ckpt.pth'))
  model.eval()
  model.to(device)

  for name, param in model.named_parameters():
    print(name, param.requires_grad)

  transform = CLIPImageProcessor.from_pretrained(backbone)
  tokenizer = CLIPTokenizer.from_pretrained(backbone)
  opt_kwargs = {}
  args = {}
  args['batch_size'] = 24
  args['num_workers'] = 8
  data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)



  ext_test_stats, task_key = data_pre.evaluate(data_loader_test, model, device, task_handler)
  print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
