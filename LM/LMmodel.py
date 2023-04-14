import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import random
from torchvision.datasets.folder import default_loader
from typing import List, Optional, Tuple


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class CLIPTextOnly(nn.Module):

    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_projection = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").text_projection
        for p in self.text_projection.parameters():
            p.requires_grad = False
        self.logit_scale = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").logit_scale

    def forward(self,
                input_ids0: Optional[torch.LongTensor] = None,
                attention_mask0: Optional[torch.Tensor] = None,
                input_ids1: Optional[torch.LongTensor] = None,
                attention_mask1: Optional[torch.Tensor] = None,
                ):
        text_outputs0 = self.text_model(
            input_ids=input_ids0,
            attention_mask=attention_mask0,
        )

        text_outputs1 = self.text_model(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        text_embeds0 = text_outputs0[1]
        text_embeds0 = self.text_projection(text_embeds0)
        text_embeds0 = text_embeds0 / text_embeds0.norm(p=2, dim=-1, keepdim=True)

        text_embeds1 = text_outputs0[1]
        text_embeds1 = self.text_projection(text_embeds1)
        text_embeds1 = text_embeds1 / text_embeds1.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds0, text_embeds1.t()) * logit_scale
        loss = clip_loss(logits_per_text)

        return loss, logits_per_text, torch.argmax(logits_per_text, dim=1)


def merge_batch_tensors_by_dict_key(batch):
    batch_tensors = {}
    for tensor_key in batch[0]:
        if isinstance(batch[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in batch])
        else:
            batch_tensors[tensor_key] = (([d[tensor_key] for d in batch]))

    return batch_tensors


class FlickrDataset(Dataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl",)
        elif split == "val":
            return (f"{task}.val.jsonl",)
        elif split == "test":
            return (f"{task}.test.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __init__(
            self, data_path, split,
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
        self.split = split

        self.text = [[]]
        for d in items:
            if d['image_id'] < len(self.text):
                self.text[d['image_id']].append(d['text_segment'])
            else:
                self.text.append([d['text_segment']])

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
        return tokens + [1] * (max_len - num_tokens), padding_mask, num_tokens  ###changed

    def _get_text_text_example(self, index: int, data: dict):
        item = self.text[index]
        random_numbers = random.sample(range(5), 2)
        text_segment = item[random_numbers[0]]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens0"] = torch.tensor(language_tokens)
        data["padding_mask0"] = torch.tensor(padding_mask)

        text_segment = item[random_numbers[1]]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens1"] = torch.tensor(language_tokens)
        data["padding_mask1"] = torch.tensor(padding_mask)
        return data

    def __getitem__(self, index: int):
        data = dict()
        self._get_text_text_example(index, data)
        return data

    def __len__(self):
        return len(self.text)


def get_dataset(split, tokenizer, batch_size, num_workers):
    #######################
    dataset_val = FlickrDataset(
        data_path='./data/flickr', split=split,
        tokenizer=tokenizer, num_max_bpe_tokens=64,
        task='flickr30k'
    )
    sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=merge_batch_tensors_by_dict_key,
    )
    return data_loader
