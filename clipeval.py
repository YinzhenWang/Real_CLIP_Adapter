import os
# pip install git+https://github.com/openai/CLIP.git
import clip
import torch
from torchvision.datasets import Caltech101,CIFAR10,CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

# Download the dataset
dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True,train=True)
#dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True,train=True)
#dataset = Caltech101(root=os.path.expanduser("~/.cache"), download=True)

# Prepare the inputs
for key in MODELS.keys():
    model, preprocess = clip.load(key, device)
    total = 0
    top1_cnt = 0
    top3_cnt = 0
    top10_cnt = 0
    with tqdm(dataset, desc=key) as loop:
        for image, class_id in loop:
            image_input = preprocess(image).unsqueeze(0).to(device)
            #caltech: dataset.categories
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            if similarity[0].topk(1)[1] == class_id:
                top1_cnt += 1
            if class_id in similarity[0].topk(3).indices.data:
                top3_cnt += 1
            if class_id in similarity[0].topk(10).indices.data:
                top10_cnt += 1
            total += 1
            loop.set_postfix(acc1=top1_cnt / total, acc3=top3_cnt / total, acc10=top10_cnt / total)

    print(key,top1_cnt / total, top3_cnt / total, top10_cnt / total)
