import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel,  CLIPTextModelWithProjection,CLIPVisionModelWithProjection
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_ as __call_trunc_normal_



def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
    
def gather_features(
        image_features,
        text_features,
):
    gathered_image_features = GatherLayer.apply(image_features)
    gathered_text_features = GatherLayer.apply(text_features)
    all_image_features = torch.cat(gathered_image_features)
    all_text_features = torch.cat(gathered_text_features)

    return all_image_features, all_text_features

# The implementation code is modified from open_clip (https://github.com/mlfoundations/open_clip.git)
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

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features
            )

            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
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


class CLIPALL(nn.Module):
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(self):
        super().__init__()
        self.backbone = "openai/clip-vit-base-patch16"
        self.cliptext = CLIPTextModelWithProjection.from_pretrained(self.backbone)
        self.clipvision = CLIPVisionModelWithProjection.from_pretrained(self.backbone)
        # self.classifer = nn.Squential(
        #     nn.Dropout(dropout),
        #     nn.Linear(input_size, input_size),
        #     torch.tanh(x),
        #     nn.Dropout(dropout),
        #     nn.Linear(input_size, 2)
        # )
        # embed_dim = 512
        # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.language_head.apply(self._init_weights)
        # self.vision_head.apply(self._init_weights)
        self.criterion = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def forward(self, inputs, **kwargs):
        text_input = {key: inputs[key] for key in ["input_ids","attention_mask"]}
        text_outputs = self.cliptext(**text_input)
        language_cls = text_outputs.text_embeds
        # language_cls = self.classifer(language_cls)

        vision_input = {key: inputs[key] for key in ["pixel_values"]}
        vision_outputs = self.clipvision(**vision_input)
        vision_cls = vision_outputs.image_embeds
        # vision_cls = self.classifer(vision_cls)

        loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
        
        output = {}
        output['vision_cls'] = vision_cls
        output['language_cls'] = language_cls
        output['loss'] = loss
        return output







