from transformers import CLIPTokenizer, CLIPTextModel, RobertaForMaskedLM, CLIPTokenizerFast, \
    DataCollatorForLanguageModeling
from LMmodel import CLIPTextMLM, get_masked_dataset
import torch
from torch.nn import DataParallel
import logging
from tqdm import tqdm
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig, LoRAConfig
from transformers.optimization import get_linear_schedule_with_warmup

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('mlmlog.txt')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
tokenizer.add_tokens(['<mask>'])

model = CLIPTextMLM()
# if torch.cuda.device_count() > 1:
#     print('start multi-gpus training')
#     model = DataParallel(model, device_ids=[2, 4, 7])
# else:
#     print('start single gpu training')
config = LoRAConfig(r=8, alpha=16)

model.cliptext.add_adapter("LoRA", config=config)
model.cliptext.set_active_adapters("LoRA")
model.cliptext.train_adapter("LoRA")
model.to(device)
for name, param in model.named_parameters():
    print(name, param.requires_grad)

num_epochs = 20
dataset = get_masked_dataset(split='train', tokenizer=tokenizer, batch_size=128, num_workers=4)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
total_steps = len(dataset) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)
for epoch in range(num_epochs):
    cnt = 0
    total = 1e-30

    with tqdm(dataset, desc='train {}'.format(epoch)) as loop:
        model.train()
        for item in loop:
            full_tokens = item["full_tokens"]
            masked_tokens = item["masked_tokens"]
            attention_mask = item["attention_mask"]
            inputs = {}
            inputs["input_ids"] = masked_tokens.to(device)
            inputs["attention_mask"] = attention_mask.to(device)

            labels = torch.where(masked_tokens == 49408, full_tokens, -100)
            x, loss = model(inputs, labels.to(device))

            for i in range(len(x)):
                mask_token_index = (masked_tokens == 49408)[i].nonzero(as_tuple=True)[0]
                predicted_token_id = x[i, mask_token_index].argmax(axis=-1)
                for j in range(len(mask_token_index)):
                    total += 1
                    cnt += (predicted_token_id[j] == labels[i][mask_token_index[j]])
                    # print(tokenizer.decode(predicted_token_id[j]), tokenizer.decode(labels[i][mask_token_index[j]]))
            optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=loss, total=total, acc=cnt / total)
    logger.info('epoch', str(epoch), 'acc', str(cnt / total))
    model.cliptext.save_adapter("./mlm_LoRA_epoch{}".format(epoch), "LoRA")
