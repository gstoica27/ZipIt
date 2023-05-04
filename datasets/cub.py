import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datetime as dt
import xml.etree.ElementTree as ET
from PIL import Image
import scipy.io
from utils import create_heldout_split

class CUB2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, base_set=None):
        self.root_og = root
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        
        if base_set is None:
            self._load_metadata()
        else:
            self.dataset = base_set
            self.targets = base_set['target'].unique()
            
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        dataset = images.merge(image_class_labels, on='img_id')
        self.dataset = dataset.merge(train_test_split, on='img_id')
        
        if self.train:
            self.dataset = self.dataset[self.dataset.is_training_img == 1]
        else:
            self.dataset = self.dataset[self.dataset.is_training_img == 0]
        self.targets = self.dataset['target'].unique()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
cub200_train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

cub200_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def cub200_train_loader():
    return CUB2011(root='/srv/datasets/cub200', train=True, transform=cub200_train_transforms)


def cub200_test_loader():
    return CUB2011(root='/srv/datasets/cub200', train=False, transform=cub200_test_transforms)


def prepare_train_loaders(config):
    return {
        'full': torch.utils.data.DataLoader(
            CUB2011(root='/srv/datasets/cub200', train=True, transform=cub200_train_transforms),
            batch_size=config['batch_size'],
            shuffle=config['shuffle_train'],
            num_workers=config['num_workers']
        )
            
    }

def prepare_test_loaders(config):
    loaders = {
        'full': torch.utils.data.DataLoader(
            CUB2011(root='/srv/datasets/cub200', train=False, transform=cub200_test_transforms),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
    }
    
    if config.get('val_fraction', 0) > 0.:
        test_set = loaders['full'].dataset
        test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        loaders['heldout_test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['heldout_val'] = torch.utils.data.DataLoader(
            val_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    return loaders

