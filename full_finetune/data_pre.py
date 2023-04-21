### the following code is a modification of https://github.com/microsoft/unilm/blob/master/beit3/ for personal use

import math
import os
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets.folder import default_loader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transformers import CLIPModel, XLMRobertaTokenizer, CLIPTokenizerFast,CLIPImageProcessor




from timm.utils import ModelEma
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer



import datetime
import io
import os
import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from timm.utils import get_state_dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._six import inf
from torchmetrics import Metric
# from tensorboardX import SummaryWriter





def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        # if torch.cuda.is_available():
        #     log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        # iterable.open_lmdb
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if torch.cuda.is_available():
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time),
                #         memory=torch.cuda.max_memory_allocated() / MB))
                # else:
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()



    def after_eval(self, **kwargs):
        raise NotImplementedError()


class RetrievalHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.metric_logger = None

    def train_batch(self, model, image, language_tokens, padding_mask, image_id):
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask)
        return {
            "loss": loss, 
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):


        inputs = {}

        inputs["pixel_values"] =  torch.squeeze(image)
        inputs["input_ids"] = language_tokens
        inputs["attention_mask"] =(1-padding_mask)



        outputs = model(**inputs)

        vision_cls = outputs.image_embeds 
        language_cls = outputs.text_embeds 


        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())

    def after_eval(self, **kwargs):
      # print(self.image_feats.length)
      # print(self.text_feats.length)
      # print(self.image_ids.length)
      
      image_feats = {}
      for feats, ids in zip(self.image_feats, self.image_ids):
          for i, _idx in enumerate(ids):
              idx = _idx.item()
              if idx not in image_feats:
                  image_feats[idx] = feats[i]
      
      tiids = torch.cat(self.image_ids, dim=0)
      iids = []
      sorted_tensors = []
      for key in sorted(image_feats.keys()):
          sorted_tensors.append(image_feats[key].view(1, -1))
          iids.append(key)

      image_cls_feats = torch.cat(sorted_tensors, dim=0)
      text_cls_feats = torch.cat(self.text_feats, dim=0)

      
      scores = image_cls_feats @ text_cls_feats.t()
      iids = torch.LongTensor(iids).to(scores.device)

      print("scores: {}".format(scores.size()))
      print("iids: {}".format(iids.size()))
      print("tiids: {}".format(tiids.size()))

      topk10 = scores.topk(10, dim=1)
      topk5 = scores.topk(5, dim=1)
      topk1 = scores.topk(1, dim=1)
      
      topk10_iids = tiids[topk10.indices]
      topk5_iids = tiids[topk5.indices]
      topk1_iids = tiids[topk1.indices]

      tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
      tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
      tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

      topk10 = scores.topk(10, dim=0)
      topk5 = scores.topk(5, dim=0)
      topk1 = scores.topk(1, dim=0)
      topk10_iids = iids[topk10.indices]
      topk5_iids = iids[topk5.indices]
      topk1_iids = iids[topk1.indices]

      ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
      ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
      ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

      eval_result = {
          "tr_r10": tr_r10.item() * 100.0, 
          "tr_r5": tr_r5.item() * 100.0, 
          "tr_r1": tr_r1.item() * 100.0, 
          "ir_r10": ir_r10.item() * 100.0, 
          "ir_r5": ir_r5.item() * 100.0, 
          "ir_r1": ir_r1.item() * 100.0, 
          "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
      }

      print('* Eval result = %s' % json.dumps(eval_result))
      return eval_result, "average_score"
    






class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = 49406
        self.eos_token_id = 49407
        self.pad_token_id = 1
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image).pixel_values

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
            # print(tokens)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [1] * (max_len - num_tokens), padding_mask, num_tokens ###changed

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body
    
class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

def build_transform(input_size):

    
    t = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

    return t

def merge_batch_tensors_by_dict_key(batch):
    batch_tensors = {}
    for tensor_key in batch[0]:
        if isinstance(batch[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in batch])
        else:
            batch_tensors[tensor_key] = (([d[tensor_key] for d in batch]))

    return batch_tensors


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    # switch to evaluation mode
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = torch.tensor(data[tensor_key]).to(device, non_blocking=True)
        handler.eval_batch(model=model, **data)

    metric_logger.synchronize_between_processes()

    return handler.after_eval()




def get_train_dataset(transform,tokenizer, batch_size, num_workers,opt_kwargs):
###################train_data_loader
  dataset_train = RetrievalDataset(
          data_path='./data/flickr', split="train", 
          transform=transform, tokenizer=tokenizer, 
          num_max_bpe_tokens=64, 
          task='flickr30k', **opt_kwargs, 
      )
#   sampler = torch.utils.data.SequentialSampler(dataset_train)
  data_loader_train = torch.utils.data.DataLoader(
          dataset_train, shuffle = True,
          batch_size=batch_size,
          num_workers=num_workers,
          drop_last=False,
          collate_fn=merge_batch_tensors_by_dict_key,
      )
  return data_loader_train

def get_test_dataset(transform,tokenizer, batch_size, num_workers,opt_kwargs):
#####test data loader
  dataset_test = RetrievalDataset(
          data_path='./data/flickr', split="test", 
          transform=transform, tokenizer=tokenizer, 
          num_max_bpe_tokens=64, 
          task='flickr30k', **opt_kwargs, 
      )
  sampler = torch.utils.data.SequentialSampler(dataset_test)
  data_loader_test = torch.utils.data.DataLoader(
          dataset_test, sampler=sampler,
          batch_size=batch_size,
          num_workers=num_workers,
          drop_last=False,
          collate_fn=merge_batch_tensors_by_dict_key,
      )
  return data_loader_test

def get_val_dataset(transform,tokenizer, batch_size, num_workers,opt_kwargs):
#######################
  dataset_val = RetrievalDataset(
          data_path='./data/flickr', split="val", 
          transform=transform, tokenizer=tokenizer, 
          num_max_bpe_tokens=64, 
          task='flickr30k', **opt_kwargs, 
      )
  sampler = torch.utils.data.SequentialSampler(dataset_val)
  data_loader_val = torch.utils.data.DataLoader(
          dataset_val, sampler=sampler,
          batch_size=batch_size,
          num_workers=num_workers,
          drop_last=False,
          collate_fn=merge_batch_tensors_by_dict_key,
      )
  return data_loader_val

backbones = ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32","openai/clip-vit-large-patch14","openai/clip-vit-large-patch14-336"]
# backbone = "openai/clip-vit-large-patch14-336"
  
  # 



if __name__ == "__main__":
  device = torch.device('cpu')#cuda
  task_handler = RetrievalHandler()
  
  backbone = "openai/clip-vit-base-patch16"

  # model = CustomCLIP(backbone)
  model = CLIPModel.from_pretrained(backbone)


  transform = CLIPImageProcessor.from_pretrained(backbone)
  tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
  opt_kwargs = {}
  args = {}
  args['batch_size'] = 2
  args['num_workers'] = 8
  data_loader_train = get_train_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  data_loader_test = get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  data_loader_val = get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
  
        
 
  for batch_idx, samples in enumerate(data_loader_train):
    print('test',batch_idx, samples)
    break

  # ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler)
  # print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
