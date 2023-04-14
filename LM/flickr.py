from LMmodel import get_dataset, CLIPTextOnly
from transformers import CLIPModel, XLMRobertaTokenizer, CLIPTokenizerFast, CLIPImageProcessor
import torch
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig,LoRAConfig
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


backbone = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizerFast.from_pretrained(backbone)

dataloader_train = get_dataset(split="train", tokenizer=tokenizer, batch_size=64, num_workers=4)
dataloader_val = get_dataset(split="val", tokenizer=tokenizer, batch_size=8, num_workers=4)

device = "cuda:4"

model = CLIPTextOnly()
config = LoRAConfig(r=8, alpha=16)
model.text_model.add_adapter("LoRA", config=config)
model.text_model.train_adapter("LoRA")
model.text_model.set_active_adapters("LoRA")

model.to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for e in range(2):
    losses = AverageMeter()
    model.train()
    with tqdm(dataloader_train, desc='train {}'.format(e)) as loop:
        for item in loop:

            loss, _, _ = model(item["language_tokens0"].to(device), item["padding_mask0"].to(device),
                               item["language_tokens1"].to(device), item["padding_mask1"].to(device))
            losses.update(loss)

            optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            optimizer.step()
            loop.set_postfix(loss=losses.avg)

model.text_model.save_adapter("./flickr_LoRA", "LoRA")