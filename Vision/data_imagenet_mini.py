import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class ImageNetMini_MIM(Dataset):
    def __init__(self, root, split, transform, mask_generator, max_len=10000):
        self.data = []
        self.img_path = []
        split_path = os.path.join(root, split)
        cnt = 0
        for root, dirs, files in os.walk(split_path):
            for file in files:
                if cnt >= max_len:
                    break
                if file.endswith('.JPEG'):
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img)
                    if len(img.shape) == 3:
                        self.data.append(img)
                        self.img_path.append(img_path)
                        cnt += 1
        self.transform = transform
        self.mask_generator = mask_generator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.mask_generator()
        try:
            pixel_values = self.transform(img).pixel_values[0]
        except:
            raise IndexError("Cannot transform data", self.img_path[index])
        return pixel_values, mask


def get_imagenet_mini(root, split, transform, mask_generator, batch_size, num_workers, max_len):
    dataset_train = ImageNetMini_MIM(root, split=split, transform=transform,
                                     mask_generator=mask_generator, max_len=max_len)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader_train


if __name__ == "__main__":
    from transformers import CLIPImageProcessor
    from mask_generator import MaskGenerator
    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mask_generator = MaskGenerator(input_size=224, mask_patch_size=16,
                                   model_patch_size=16, mask_ratio=0.4)
    data_loader = get_imagenet_mini('../dataset/data/imagenet', 'val', transform, mask_generator, 128, 8, max_len=4000)
    # check
    for batch in data_loader:
        img, mask = batch
        print(img.shape, mask.shape)
