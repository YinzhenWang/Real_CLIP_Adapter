import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CLIPVisionClassification(nn.Module):

    def __init__(self, input_size, class_num, dropout):
        super().__init__()
        self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.classifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, class_num)
        )

    def forward(self, inputs, **kwargs):
        outputs = self.clipvision(**inputs)
        x = outputs.pooler_output
        x = self.classifer(x)
        return x


class CLIPVisionMasked(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.vision_model = self.clipvision.vision_model

        self.vision_embedding = self.clipvision.vision_model.embeddings
        self.patch_embedding = self.clipvision.vision_model.embeddings.patch_embedding
        self.class_embedding = self.clipvision.vision_model.embeddings.class_embedding
        self.position_embedding = self.clipvision.vision_model.embeddings.position_embedding

        self.embed_dim = self.clipvision.vision_model.config.hidden_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)

    def forward(self, pixel_values, mask):
        if mask is None:
            raise ValueError("The mask is None")

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        B, L, _, = patch_embeds.shape

        # Pay attention here
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        patch_embeds = patch_embeds * (1 - w) + mask_token * w

        class_embeds = self.class_embedding.expand(B, 1, -1)
        embeddings = torch.cat((class_embeds, patch_embeds), dim=1)
        embeddings = embeddings + self.position_embedding(self.vision_embedding.position_ids)

        embeddings = self.vision_model.pre_layrnorm(embeddings)
        embeddings = self.pos_drop(embeddings)

        encoder_outputs = self.vision_model.encoder(inputs_embeds=embeddings)
        last_hidden_state = encoder_outputs[0]
        outputs = last_hidden_state[:, 1:, :]
        outputs = self.vision_model.post_layernorm(outputs)

        B, L, C = outputs.shape
        H = W = int(L ** 0.5)
        outputs = outputs.permute(0, 2, 1).reshape(B, C, H, W)

        return outputs


class MIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.encoder.embed_dim,
        #         out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
        #     nn.PixelShuffle(self.encoder_stride),
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.embed_dim,
                out_channels=self.encoder.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.encoder.embed_dim,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        return x_rec
