import os
import pdb
import clip
import torch
import torchvision
import random
from copy import deepcopy

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model_merger import ModelMerge


def fix_state_dict_maps(path, device):
    state_dict = torch.load(path, map_location=device)
    new_sd = {}
    for key, val in state_dict.items():
        key = key.replace('module.', '')
        new_sd[key] = val
    return new_sd
    
    
if __name__ == "__main__":
    config_name = 'multidataset_resnet50'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = 'stanford_dogs'
    model_name = 'resnet50'
    weights = 'daniel_ckpt1'
    model = torchvision.models.resnet50().to(device)
    # Daniel's Models
    chkpt_path = '/srv/share2/gstoica3/checkpoints/imagenet/resnet50_imagenet1k_1.pt'
    ckpt_sd = fix_state_dict_maps(chkpt_path, device)
    model.load_state_dict(ckpt_sd)
    
    loaders = prepare_data(config={'name': dataset_name})
    seed = 17
    save_dir = f'/srv/share2/gstoica3/checkpoints/multiset/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_from_{weights.lower()}_v0.pth.tar')
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    print(f'Starting to Train on {dataset_name}')
    trained_model = train_logits(
        model, 
        train_loader=loaders['train']['full'], 
        test_loader=loaders['test']['full']
    )
    print('Saving Model...')
    save_model(trained_model, save_path)
    print('Model Saved!')