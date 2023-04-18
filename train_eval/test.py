import torch
from transformers import CLIPTokenizerFast, CLIPImageProcessor

from data_pre import RetrievalHandler, get_test_dataset, evaluate
from model import CLIPALL

backbones = ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14",
             "openai/clip-vit-large-patch14-336"]


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task_handler = RetrievalHandler()

    backbone = "openai/clip-vit-base-patch16"

    filedir = "./output_adapter"
    text_configname = "mam"

    model = CLIPALL(backbone)
    model.clipvision.load_adapter(filedir + "_v")
    model.clipvision.set_active_adapters("LoRA")
    model.cliptext.load_adapter(filedir + "_t")
    model.cliptext.set_active_adapters(text_configname)
    model.load_state_dict(torch.load(filedir + '_v/ckpt.pth'))
    model.eval()
    model.to(device)

    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 24
    args['num_workers'] = 8
    data_loader_test = get_test_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                        opt_kwargs)

    ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler)
    print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
