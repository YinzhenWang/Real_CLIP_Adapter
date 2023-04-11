import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTModel, ViTFeatureExtractor, ViTConfig, AdapterType, AdapterConfig, AutoConfig, AutoModel
from transformers.adapters import ViTAdapterModel, AdapterConfig, MAMConfig, UniPELTConfig

# GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# define image preprocessing
transform_train = transforms.Compose([
    # transforms.RandomCrop(224, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

# Training and Validation Set
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(num_train * 0.8)  # 80% for training，20% for testing
train_idx, val_idx = indices[:split], indices[split:]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx) # split train and validation data


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=val_sampler, num_workers=4)

print('Train on CIFAR 100.')
print('train data:', len(train_sampler), ', validation data:', len(val_sampler))
print('train data batch:', len(train_loader), ', validation data batch:', len(val_loader))

# add ViT
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
model = ViTAdapterModel.from_pretrained('google/vit-base-patch16-224', config=config)


# add adapter for image classification
# use tanh
# config = {'head_type': 'image_classification', 'num_labels': 10, 'layers': 1, 'activation_function': 'tanh', 'multilabel': False, 
#           'label2id': train_dataset.class_to_idx, 
#           'use_pooler': False, 'bias': True, 'mh_adapter': True, 'output_adapter': True, 'reduction_factor': 16, 'non_linearity': 'tanh'}
config = MAMConfig()
model.add_adapter("cifar100_adapter", config=config)

model.train_adapter("cifar100_adapter")
model.set_active_adapters("cifar100_adapter")

# add visual head
model.add_image_classification_head("cifar100_adapter", num_labels=100, 
                                    id2label=train_dataset.class_to_idx)
model = model.to(device)

# add ViT pretrained feature extractor
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0
    train_corrects = 0
    val_corrects = 0

    # training
    model.train()
    for batch_idx, (train_x, train_y) in tqdm(enumerate(train_loader)):
        # pass the feature_extractor
        # train_x_features = []
        # for i in train_x:
        #     i_feature = feature_extractor(images=i, return_tensors="pt").pixel_values
        #     train_x_features.append(i_feature) # need pass image one by one
        # train_x_new = torch.cat(train_x_features, dim=0) # stack
        # # pass the adapter
        # train_x, train_y = train_x_new.to(device), train_y.to(device)
        train_x, train_y = train_x.to(device), train_y.to(device)
        p = model(train_x, adapter_names=["cifar100_adapter"]).logits
        loss = criterion(p, train_y)
        pre_lab = torch.argmax(p, axis=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # model.train_adapter(["cifar100_adapter"])

        train_loss_epoch += loss.item() * train_x.size(0)
        train_corrects += torch.sum(pre_lab == train_y.data)
        if batch_idx % 200 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')
            # print(train_loss_epoch, train_corrects, len(train_sampler))
    # calculate loss and accuracy for one epoch
    train_loss = train_loss_epoch / len(train_sampler)
    train_acc = train_corrects.double() / len(train_sampler)
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_acc:.2%}')

    # validate
    model.eval()
    with torch.no_grad():
        for batch_idx, (val_x, val_y) in enumerate(val_loader):
            # pass the feature_extractor
            # val_x_features = []
            # for i in val_x:
            #     i_feature = feature_extractor(images=i, return_tensors="pt").pixel_values
            #     val_x_features.append(i_feature)
            # val_x_new = torch.cat(val_x_features, dim=0) 
            # pass the adapter
            val_x, val_y = val_x.to(device), val_y.to(device)
            p = model(val_x, adapter_names=["cifar100_adapter"]).logits
            loss = criterion(p, val_y)
            pre_lab = torch.argmax(p, axis=1)
            val_loss_epoch += loss.item() * val_x.size(0)
            val_corrects += torch.sum(pre_lab == val_y.data)
    # calculate loss and accuracy for one epoch
    val_loss = val_loss_epoch / len(val_sampler)
    val_acc = val_corrects.double() / len(val_sampler)
    print(f'Epoch: {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Acc: {val_acc:.2%}')


# Testing set
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

test_loss_epoch = 0.0
test_corrects = 0

# testing
model.eval()
with torch.no_grad():
    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        # pass the feature_extractor
        test_x_features = []
        # for i in test_x:
        #     i_feature = feature_extractor(images=i, return_tensors="pt").pixel_values
        #     test_x_features.append(i_feature)
        # test_x_new = torch.cat(test_x_features, dim=0) 
        # pass the adapter
        test_x, test_y = test_x.to(device), test_y.to(device)
        p = model(test_x, adapter_names=["cifar100_adapter"]).logits
        loss = criterion(p, test_y)
        pre_lab = torch.argmax(p, axis=1)
        test_loss_epoch += loss.item() * test_x.size(0)
        test_corrects += torch.sum(pre_lab == test_y.data)
    # calculate loss and accuracy for whole test set
    test_loss = test_loss_epoch / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Loss: {test_loss}, Test Acc: {test_acc:.2%}')