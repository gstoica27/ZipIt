import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import pdb

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

################################################# Global Variables #################################################

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))

####################################################################################################################

def prepare_train_loaders(config):
    if 'no_transform' in config:
        train_transform = T.Compose([T.ToTensor(), normalize])
    else:
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])

    train_dset = config['wrapper'](root=config['dir'], train=True, download=True, transform=train_transform)
    loaders = {'full': torch.utils.data.DataLoader(train_dset, batch_size=config['batch_size'], shuffle=config['shuffle_train'], num_workers=config['num_workers'])}
    
    if 'class_splits' in config:
        loaders['splits'] = []
        grouped_class_indices = np.zeros(config['num_classes'], dtype=int)
        for i, splits in enumerate(config['class_splits']):
            valid_examples = [i for i, label in tqdm(enumerate(train_dset.targets)) if label in splits]
            data_subset = torch.utils.data.Subset(train_dset, valid_examples)
            loaders['splits'].append(torch.utils.data.DataLoader(
                data_subset, batch_size=config['batch_size'], shuffle=config['shuffle_train'], num_workers=config['num_workers']
            ))
            grouped_class_indices[splits] = np.arange(len(splits))

        loaders["label_remapping"] = torch.from_numpy(grouped_class_indices)
        loaders['class_splits'] = config['class_splits']
    
    return loaders

def prepare_test_loaders(config):
    test_transform = T.Compose([T.ToTensor(), normalize])
    test_dset = config['wrapper'](root=config['dir'], train=False, download=True, transform=test_transform)
    loaders = {'full': torch.utils.data.DataLoader(test_dset, batch_size=config['batch_size'], shuffle=config["shuffle_test"], num_workers=config['num_workers'])}
    
    if 'class_splits' in config:
        loaders['splits'] = []
        for i, splits in enumerate(config['class_splits']):
            valid_examples = [i for i, label in tqdm(enumerate(test_dset.targets)) if label in splits]
            data_subset = torch.utils.data.Subset(test_dset, valid_examples)
            loaders['splits'].append(torch.utils.data.DataLoader(
                data_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
            ))
            
    loaders['class_names'] = test_dset.classes
    return loaders
