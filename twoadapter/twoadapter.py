import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer
from transformers.adapters import AdapterConfig, MAMConfig, UniPELTConfig
from model import CLIPALL
import data_pre
from transformers import CLIPModel, XLMRobertaTokenizer, CLIPTokenizerFast,CLIPImageProcessor
import logging
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig





if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    backbone = "openai/clip-vit-base-patch16"

    # model = CustomCLIP(backbone)
    # model = CLIPModel.from_pretrained(backbone)


    transform = CLIPImageProcessor.from_pretrained(backbone)
    tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
    opt_kwargs = {}
    args = {}
    args['batch_size'] = 16
    args['num_workers'] = 8
    data_loader_train = data_pre.get_train_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    data_loader_val = data_pre.get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)












    model = CLIPALL()
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

    model.cliptext.add_adapter("adapter_t", config=config)
    model.cliptext.train_adapter("adapter_t")
    model.cliptext.set_active_adapters("adapter_t")

    model.clipvision.add_adapter("adapter_v", config=config)
    model.clipvision.train_adapter("adapter_v")
    model.clipvision.set_active_adapters("adapter_v")


    model.to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    criterion = model.criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)



    for _ in range(1):

        model.train()
        with tqdm(data_loader_train, desc='train {}'.format(_)) as loop:
            for x in loop:

                for key,value in x.items():
                    x[key] = torch.tensor(value).to(device, non_blocking=True)
                
                inputs = {}
                inputs["pixel_values"] =  torch.squeeze(x['image'])
                inputs["input_ids"] = x['language_tokens']
                inputs["attention_mask"] =(1-x['padding_mask'])
                outputs = model(inputs)
                vision_cls = outputs['vision_cls']
                language_cls = outputs['language_cls']

                loss,logit1,logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss)
        logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_,configname,loss))

        model.eval()
        with tqdm(data_loader_val, desc='test {}'.format(_)) as loop:
            for x in loop:

                for key,value in x.items():
                    x[key] = torch.tensor(value).to(device, non_blocking=True)
                inputs = {}
                inputs["pixel_values"] =  torch.squeeze(x['image'])
                inputs["input_ids"] = x['language_tokens']
                inputs["attention_mask"] =(1-x['padding_mask'])
                outputs = model(inputs)
                vision_cls = outputs['vision_cls']
                language_cls = outputs['language_cls']

                loss,logit1,logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                loop.set_postfix(loss=loss)
        logger.info('val_epoch{}_adapter_{}_acc_{}'.format(_, configname, loss))



    model.cliptext.save_adapter("./adaptert", "adapter_t")
    model.clipvision.save_adapter("./adapterv", "adapter_v")