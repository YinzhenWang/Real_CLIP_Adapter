import torch
import numpy as np
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
import torch.nn.functional as F




if __name__ == "__main__":
    output_dir = "./output_adapter"
    checkpoint_dir = "./lora_ckpt"
    log_path = "log.txt"
    text_configname = "mam"


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
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
    # data_loader_test = data_pre.get_test_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)
    # data_loader_val = data_pre.get_val_dataset(transform,tokenizer, args['batch_size'], args['num_workers'],opt_kwargs)




    model = CLIPALL(backbone)
    configs = {
        "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
        "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
        "LoRA": LoRAConfig(r=8, alpha=16),
        "IA3": IA3Config(),
        "mam": MAMConfig(),
        "unipelt": UniPELTConfig()
    }
    
    # config = configs[configname]



    model.clipvision.load_adapter(checkpoint_dir)
    model.clipvision.set_active_adapters("LoRA")
    model.clipvision.train_adapter("LoRA")


    model.cliptext.add_adapter(text_configname, config=configs[text_configname])
    model.cliptext.train_adapter(text_configname)
    model.cliptext.set_active_adapters(text_configname)


    for name, param in model.named_parameters():
        param.requires_grad = False
        if "LoRA" in name or "mam" in name:
            param.requires_grad = True
        print(name, param.requires_grad)


    criterion = model.criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)


    model.to(device)
    for _ in range(20):

        model.train()
        with tqdm(data_loader_train, desc='train {}'.format(_)) as loop:
            # cnt = 0
            for x in loop:
                # cnt+=1
                # if cnt%4500 == 0.0:
                #     logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_,text_configname,loss))
                #     model.clipvision.save_adapter(output_dir+"_v", "LoRA")
                #     model.cliptext.save_adapter(output_dir+"_t", "mam")
                #     torch.save(model.state_dict(), output_dir+'_v/ckpt.pth')
                for key,value in x.items():
                    x[key] = torch.tensor(np.array(value)).to(device, non_blocking=True)

                inputs = {}
                inputs["pixel_values"] =  torch.squeeze(x['pixel_values'])
                inputs["input_ids"] = x['input_ids']
                inputs["attention_mask"] =(1-x['attention_mask'])
                outputs = model(**inputs)
                vision_cls = outputs.image_embeds
                language_cls = outputs.text_embeds

                loss,logit1,logit2 = criterion(vision_cls, language_cls, model.logit_scale)

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss)
                logger.info('train_epoch{}_adapter_{}_acc_{}'.format(_,text_configname,loss))
        model.clipvision.save_adapter(output_dir+"_v", "LoRA")
        model.cliptext.save_adapter(output_dir+"_t", "mam")
        torch.save(model.state_dict(), output_dir+'_v/ckpt.pth')
