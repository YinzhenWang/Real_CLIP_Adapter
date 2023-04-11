import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CLIPVisionClassification(nn.Module):

    def __init__(self, input_size, class_num, dropout):
        super().__init__()
        self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
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

'''
class CLIPVisionDataset(Dataset):
    def __init__(self, dataset, processor,image_path = None, label_path = None):
        self.images = []
        self.labels = []
        i = 0
        for image, class_id in tqdm(dataset):
            self.images.append(processor(images=image, return_tensors="pt")['pixel_values'][0])
            self.labels.append(torch.tensor([class_id]))
            i += 1
            if i>5:
                break
            print(processor(images=image, return_tensors="pt"))
        torch.save(torch.stack(self.images),"./cifar_image.pt")
        torch.save(torch.stack(self.labels), "./cifar_label.pt")

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
'''