import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from segment_anything.modeling import TwoWayTransformer
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPVisionModel, CLIPTextModel
from torch import Tensor


class CLIPOutput:
    def __init__(self, loss, text_embeds, image_embeds, logits_per_image=None, logits_per_text=None):
        self.loss = loss
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds
        self.logits_per_image = logits_per_image
        self.logits_per_text = logits_per_text


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

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cliptext = CLIPTextModelWithProjection.from_pretrained(self.backbone)
        self.clipvision = CLIPVisionModelWithProjection.from_pretrained(self.backbone)
        num_heads = self.clipvision.vision_model.config.num_attention_heads
        embed_dim = 512
        self.cross_transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=embed_dim,  # prompt_embed_dim, #the channel dimension for the input embeddings
            mlp_dim=2048,
            num_heads=8,
            attention_downsample_rate=1,
        )
        # self.position_ids = self.clipvision.vision_model.embeddings.position_ids
        self.register_buffer("position_ids", torch.arange(197).expand((1, -1)))
        self.position_embedding = nn.Embedding(197, embed_dim)
        self.mlp_vision = nn.Linear(768, embed_dim, bias=False)
        self.mlp_text = nn.Linear(512, embed_dim, bias=False)

        self.criterion = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_ids, pixel_values, attention_mask, image_id=None):

        text_input = {}
        text_input['input_ids'] = input_ids
        text_input['attention_mask'] = attention_mask
        text_outputs = self.cliptext(**text_input)

        vision_input = {}
        vision_input['pixel_values'] = pixel_values
        vision_outputs = self.clipvision(**vision_input)

        text_embeds = text_outputs.last_hidden_state
        text_feat = self.mlp_text(text_embeds)

        image_embeds = vision_outputs.last_hidden_state
        image_feat = self.mlp_vision(image_embeds)

        # Calculate contrastive loss
        image_feat = image_feat[:, 0, :]
        text_feat = text_feat[
            torch.arange(text_feat.shape[0], device=text_feat.device),
            input_ids.to(dtype=torch.int, device=text_feat.device).argmax(dim=-1),
        ]
        image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

        loss_itc, logits_per_image, logits_per_text = self.criterion(image_feat, text_feat, self.logit_scale.exp())
        score = image_feat @ text_feat

        # Cross attention fusion
        src = image_feat.permute(0, 2, 1)
        b, c, hw = src.shape
        src = torch.reshape(src, (b, c, hw, 1))

        image_pe = self.position_embedding(self.position_ids)
        pos_src = torch.repeat_interleave(image_pe, text_feat.shape[0], dim=0)
        pos_src = pos_src.permute(0, 2, 1)
        pos_src = torch.reshape(pos_src, (b, c, hw, 1))

        hs, src = self.cross_transformer(src, pos_src, text_feat)
        # print("hs ",hs.shape) #[bs, 64, 256]
        # print("src ",src.shape) #[bs, 197, 256]

        output_pos = torch.mul(hs, src)

        with torch.no_grad():
            bs = image_feat.size(0)
            weights_i2t = F.softmax(score + 1e-4, dim=1)
            weights_t2i = F.softmax(score + 1e-4, dim=0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        image_embeds = src[:, 0, :]
        text_embeds = hs[
            torch.arange(hs.shape[0], device=hs.device),
            input_ids.to(dtype=torch.int, device=hs.device).argmax(dim=-1),
        ]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        loss, logits_per_image, logits_per_text = self.criterion(image_embeds, text_embeds, self.logit_scale.exp())

        output = CLIPOutput(
            loss=loss,
            text_embeds=text_embeds,
            image_embeds=image_embeds
        )
        return output


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    Borrow from SAM.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttentionDecoderLayer, self).__init__()
        self.self_attn = Attention(hidden_size, num_heads)
        self.cross_attn = Attention(hidden_size, num_heads)
        self.text_attn = nn.MultiheadAttention
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, queries, keys, query_pe, key_pe):
        # Self attention
        q = queries + query_pe
        self_attn_out = self.self_attn(q, q, queries)
        self_attn_out = queries + self_attn_out
        self_attn_out_norm = self.norm1(self_attn_out)

        # Cross attention
        q = self_attn_out_norm + query_pe
        k = keys + key_pe
        cross_attn_out = self.cross_attn(q, k, keys)
        cross_attn_out = queries + cross_attn_out
        cross_attn_out_norm = self.norm2(cross_attn_out)

        # MLP
        mlp_output = self.feed_forward(cross_attn_out_norm)
        mlp_output = mlp_output + cross_attn_out_norm
        mlp_output_norm = self.norm3(mlp_output)

        return mlp_output_norm


class CLIPAll_SimpleCrossAttn(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cliptext = CLIPTextModelWithProjection.from_pretrained(self.backbone)
        self.clipvision = CLIPVisionModelWithProjection.from_pretrained(self.backbone)
        embed_dim = 512

        self.mlp_vision = nn.Linear(self.clipvision.vision_model.config.hidden_size, embed_dim, bias=False)
        self.mlp_text = nn.Linear(self.cliptext.text_model.config.hidden_size, embed_dim, bias=False)

        self.register_buffer("image_position_ids", torch.arange(197).expand((1, -1)))
        self.image_position_embedding = nn.Embedding(197, embed_dim)

        self.register_buffer("text_position_ids", torch.arange(77).expand((1, -1)))
        self.text_position_embedding = nn.Embedding(77, embed_dim)

        self.image_cross_transformer = CrossAttentionDecoderLayer(hidden_size=embed_dim, num_heads=8)
        self.text_cross_transformer = CrossAttentionDecoderLayer(hidden_size=embed_dim, num_heads=8)

        self.criterion = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.itm_head = nn.Linear(embed_dim, 2)

    def forward(self, input_ids, pixel_values, attention_mask, image_id=None):
        text_input = {}
        text_input['input_ids'] = input_ids
        text_input['attention_mask'] = attention_mask
        text_outputs = self.cliptext(**text_input)

        vision_input = {}
        vision_input['pixel_values'] = pixel_values
        vision_outputs = self.clipvision(**vision_input)

        image_embeds = self.mlp_vision(vision_outputs.last_hidden_state)
        image_feat = image_embeds[:, 0, :]
        image_feat = F.normalize(image_feat, dim=-1)

        text_embeds = self.mlp_text(text_outputs.last_hidden_state)
        text_feat = text_embeds[
            torch.arange(text_embeds.shape[0], device=text_embeds.device),
            input_ids.to(dtype=torch.int, device=text_embeds.device).argmax(dim=-1),
        ]
        text_feat = F.normalize(text_feat, dim=-1)

        # Calculate contrastive loss
        loss_itc, logits_per_image, logits_per_text = self.criterion(image_feat, text_feat, self.logit_scale.exp())

        # Cross modal fusion
        image_pe = self.image_position_embedding(self.image_position_ids)
        image_pe = torch.repeat_interleave(image_pe, image_embeds.shape[0], dim=0)
        image_pe = image_pe[:, :image_embeds.size(1), :]

        text_pe = self.text_position_embedding(self.text_position_ids)
        text_pe = torch.repeat_interleave(text_pe, text_embeds.shape[0], dim=0)
        text_pe = text_pe[:, :text_embeds.size(1), :]

        image_embeds_pos = self.image_cross_transformer(image_embeds, text_embeds, image_pe, text_pe)
        image_feat_pos = image_embeds_pos[:, 0, :]
        text_embeds_pos = self.text_cross_transformer(text_embeds, image_embeds, text_pe, image_pe)
        text_feat_pos = text_embeds_pos[
            torch.arange(text_embeds_pos.shape[0], device=text_embeds_pos.device),
            input_ids.to(dtype=torch.int, device=text_embeds_pos.device).argmax(dim=-1),
        ]

        with torch.no_grad():
            batch_size = image_feat.size(0)
            weights_i2t = F.softmax(logits_per_image + 1e-4, dim=1)
            weights_t2i = F.softmax(logits_per_text + 1e-4, dim=1)

        # Select a negative image for each text
        image_embeds_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # Select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        input_ids_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(attention_mask[neg_idx])
            input_ids_neg.append(input_ids[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        input_ids_neg = torch.stack(input_ids_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([attention_mask, text_atts_neg], dim=0)
        input_ids_all = torch.cat([input_ids, input_ids_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)

        image_pe = self.image_position_embedding(self.image_position_ids)
        image_pe = torch.repeat_interleave(image_pe, image_embeds_all.shape[0], dim=0)
        image_pe = image_pe[:, :image_embeds_all.size(1), :]

        text_pe = self.text_position_embedding(self.text_position_ids)
        text_pe = torch.repeat_interleave(text_pe, text_embeds_all.shape[0], dim=0)
        text_pe = text_pe[:, :text_embeds_all.size(1), :]

        image_embeds_neg = self.image_cross_transformer(image_embeds_all, text_embeds_all, image_pe, text_pe)
        image_feat_neg = image_embeds_neg[:, 0, :]
        text_embeds_neg = self.text_cross_transformer(text_embeds_all, image_embeds_all, text_pe, image_pe)
        text_feat_neg = text_embeds_neg[
            torch.arange(text_embeds_neg.shape[0], device=text_embeds_neg.device),
            input_ids_all.to(dtype=torch.int, device=text_embeds_neg.device).argmax(dim=-1),
        ]

        vl_embeds_pos = torch.mul(image_feat_pos, text_feat_pos)
        vl_embeds_neg = torch.mul(image_feat_neg, text_feat_neg)

        vl_embeds = torch.cat([vl_embeds_pos, vl_embeds_neg], dim=0)
        vl_output = self.itm_head(vl_embeds)

        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                               dim=0).to(vl_output.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        loss = loss_itm + loss_itc

        output = CLIPOutput(
                loss=loss,
                text_embeds=text_feat,
                image_embeds=image_feat,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text
            )

        return output
