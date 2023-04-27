import logging

import torch
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from timm.utils import AverageMeter
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
from transformers import CLIPTextModel, CLIPVisionModel
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig

import data_pre
from train_eval.model import ClipLoss

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('cleanlog.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    task_handler = data_pre.RetrievalHandler()
    device = "cuda:7"

    backbone = "openai/clip-vit-base-patch16"

    model = CLIPModel.from_pretrained(backbone).to(device)
    model.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    model.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

    configs = {
        "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
        "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
        "LoRA": LoRAConfig(r=8, alpha=16),
        "IA3": IA3Config(),
        "mam": MAMConfig(),
        "unipelt": UniPELTConfig()
    }
    configname = "mam"
    config = configs[configname]
    #
    # model.vision_model.add_adapter("adapter_v", config=config)
    # model.vision_model.train_adapter("adapter_v")
    # model.vision_model.set_active_adapters("adapter_v")

    for param in model.parameters():
        param.requires_grad = False

    # model.vision_model.load_adapter("./vadapter")
    model.vision_model.add_adapter("LoRA_v", configs["LoRA"])
    model.vision_model.train_adapter("LoRA_v")
    model.vision_model.set_active_adapters("LoRA_v")

    model.text_model.load_adapter("./LM/cleaned_mam_epoch19")
    model.text_model.set_active_adapters("mam")
    model.text_model.add_adapter("LoRA_t",configs["LoRA"])
    model.text_model.train_adapter("LoRA_t")
    model.text_model.set_active_adapters("LoRA_t")

    for name, param in model.text_model.named_parameters():
        print(name, param.requires_grad)

    # Wrap model in DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        logger.info('Multi-GPU training.')
        print('Start multi-GPU training.')
        model = DataParallel(model)
        data_parallel = True
    else:
        logger.info('Single GPU training.')
        print('Start single GPU training.')
        data_parallel = False

    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 32
    args['num_workers'] = 8
    data_loader_train = data_pre.get_train_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                                   opt_kwargs)
    data_loader_test = data_pre.get_test_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                                 opt_kwargs)
    data_loader_val = data_pre.get_val_dataset(transform, tokenizer, args['batch_size'], args['num_workers'],
                                               opt_kwargs)

    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5,weight_decay=0.01)

    criterion = ClipLoss()

    for _ in range(10):
        finetune_loss = AverageMeter()
        model.train()
        with tqdm(data_loader_train, desc='line2 train {}'.format(_)) as loop:
            for x in loop:

                for key, value in x.items():
                    x[key] = torch.tensor(value).to(device, non_blocking=True)

                inputs = {}
                inputs["pixel_values"] = torch.squeeze(x['image'])
                inputs["input_ids"] = x['language_tokens']
                inputs["attention_mask"] = (1 - x['padding_mask'])
                outputs = model(input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                attention_mask=inputs["attention_mask"],
                                return_loss=True)

                vision_cls = outputs.image_embeds
                language_cls = outputs.text_embeds

                if data_parallel:
                    loss, logit1, logit2 = criterion(vision_cls, language_cls, model.module.logit_scale)
                else:
                    loss, logit1, logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                # loss = outputs.loss

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()

                finetune_loss.update(loss.item(), inputs["pixel_values"].size(0))
                loop.set_postfix(loss=finetune_loss.avg)
        logger.info('train_epoch{}_adapter_{}_loss_{}'.format(_, "none", finetune_loss.avg))

        model.eval()
        ext_test_stats, task_key = data_pre.evaluate(data_loader_test, model, device, task_handler)
        print(
            f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
        logger.info(
            f"all: Epoch{_}: Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")

        # model.text_model.save_adapter("./two", "LoRA_t")
        # model.vision_model.save_adapter("./two", "LoRA_v")
