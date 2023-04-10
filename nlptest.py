import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AlbertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers.adapters import AdapterConfig,MAMConfig,UniPELTConfig



device = "cuda:3"

train_dataset = load_dataset("glue", "mrpc", split="train")
test_dataset = load_dataset("glue", "mrpc", split="test")
'''
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased").to(device)'''

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
config = MAMConfig()
model.add_adapter("mam_adapter", config=config)
print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
model.train_adapter("mam_adapter")
model.set_active_adapters("mam_adapter")
model.to(device)
'''
for name, param in model.named_parameters():
    print(name)
    if '11' in name or '10' in name or 'dense' in name:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False
'''


def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")


print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))

train_dataset = train_dataset.map(encode, batched=True)
test_dataset = test_dataset.map(encode, batched=True)
train_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5,weight_decay=0.1)

cnt = 0
total = 0
for item in test_dataloader:
    input_ids = item['input_ids'].to(device)
    attention_mask = item['attention_mask'].to(device)
    # token_type_ids = item['token_type_ids'].to(device)
    labels = item['label'].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, labels=labels)
        pre = torch.argmax(out.logits, dim=1)
        for i in range(len(labels)):
            total += 1
            cnt += (labels[i] == pre[i])
print("init", cnt, total, cnt / total)

for _ in range(10):
    cnt = 0
    total = 0

    model.train()
    for item in train_dataloader:
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        #token_type_ids = item['token_type_ids'].to(device)
        labels = item['label'].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask,\
                output_hidden_states = True, labels = labels)
        loss = out.loss
        print(loss)
        optimizer.zero_grad()
        if loss != 0:
            loss.backward()
        optimizer.step()

    model.eval()
    for item in test_dataloader:
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        # token_type_ids = item['token_type_ids'].to(device)
        labels = item['label'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, \
                        output_hidden_states=True, labels=labels)
            pre = torch.argmax(out.logits, dim=1)
            for i in range(len(labels)):
                total += 1
                cnt += (labels[i] == pre[i])
    print("epoch",_ ,": ",cnt, total, cnt / total)
