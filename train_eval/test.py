
import math
import os
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_pre
from transformers import CLIPModel,  CLIPTokenizerFast,CLIPImageProcessor
from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
from model import CLIPALL


backbones = ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32","openai/clip-vit-large-patch14","openai/clip-vit-large-patch14-336"]


if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
  task_handler = data_pre.RetrievalHandler()

  backbone = "openai/clip-vit-base-patch16"
  
  filedir = "./output_adapter"
  text_configname = "mam"



  model = CLIPALL(backbone)
  model.clipvision.load_adapter(filedir+"_v")
  model.clipvision.set_active_adapters("LoRA")
  model.cliptext.load_adapter(filedir+"_t")
  model.cliptext.set_active_adapters(text_configname)
  model.load_state_dict(torch.load(filedir+'_v/ckpt.pth'))
  model.eval()
  model.to(device)

  transform = CLIPImageProcessor.from_pretrained(backbone)
  tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
  opt_kwargs = {}
  args = {}
  args['batch_size'] = 24
  args['num_workers'] = 8
  data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)



  ext_test_stats, task_key = data_pre.evaluate(data_loader_test, model, device, task_handler)
  print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
