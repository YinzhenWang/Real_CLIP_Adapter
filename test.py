from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel
from transformers.adapters import AdapterConfig,MAMConfig,UniPELTConfig
vconfig = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
config = UniPELTConfig()
print(config.to_dict())
#config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(model.vision_model)
inputs = processor(
    text=["a photo of a cat", "a photo of a dog","a photo of wjm"], images=image, return_tensors="pt", padding=True
)
model.text_model.add_adapter("mam_adapter", config=config)
model.text_model.train_adapter("mam_adapter")
model.text_model.set_active_adapters("mam_adapter")

model.vision_model.add_adapter("wjm", config=config)
model.vision_model.train_adapter("wjm")
model.vision_model.set_active_adapters("wjm")

print(model.vision_model)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)