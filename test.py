from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from transformers.adapters import AdapterConfig

config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(model.text_model)
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)
model.add_adapter("bottleneck_adapter", config=config)
model.train_adapter("bottleneck_adapter")
model.set_active_adapters("bottleneck_adapter")
print(model.text_model)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)