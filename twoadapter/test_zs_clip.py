
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


backbones = ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32","openai/clip-vit-large-patch14","openai/clip-vit-large-patch14-336"]
# backbone = "openai/clip-vit-large-patch14-336"
  
  # 



if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
  task_handler = data_pre.RetrievalHandler()
  
  backbone = "openai/clip-vit-base-patch16"

  # model = CustomCLIP(backbone)
  model = CLIPModel.from_pretrained(backbone)


  transform = CLIPImageProcessor.from_pretrained(backbone)
  tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
  opt_kwargs = {}
  args = {}
  args['batch_size'] = 24
  args['num_workers'] = 8
  # data_loader_train = data_pre.get_train_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  # data_loader_val = data_pre.get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  
        
 
  # for batch_idx, samples in enumerate(data_loader_train):
  #   print('test',batch_idx, samples)
  #   break

  
  ext_test_stats, task_key = data_pre.evaluate(data_loader_test, model, device, task_handler)
  print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
