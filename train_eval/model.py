import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel,  CLIPTextModelWithProjection,CLIPVisionModelWithProjection
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


class CLIPOutput:
    def __init__(self,loss, text_embeds, image_embeds):
        self.loss = loss
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds


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

    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
        self.clip = CLIPModel.from_pretrained(self.backbone)
        self.cliptext = self.clip.text_model
        self.clipvision = self.clip.vision_model
        self.text_projection = self.clip.text_projection
        self.visual_projection = self.clip.visual_projection
        self.criterion = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def forward(self, input_ids,pixel_values,attention_mask,image_id = None):

        text_input = {}
        text_input['input_ids'] = input_ids
        text_input['attention_mask'] = attention_mask
        text_outputs = self.cliptext(**text_input)
        language_cls = self.text_projection(text_outputs[1])
        text_embeds = language_cls

        vision_input = {}
        vision_input['pixel_values'] = pixel_values
        vision_outputs = self.clipvision(**vision_input)
        vision_cls = self.visual_projection(vision_outputs[1])
        image_embeds = vision_cls
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        #print(image_embeds)
        loss, logits_per_image, logits_per_text = self.criterion(vision_cls, language_cls, self.logit_scale.exp())
        
        output = CLIPOutput(
            loss=loss,
            text_embeds=text_embeds,
            image_embeds=image_embeds
        )
        return output















