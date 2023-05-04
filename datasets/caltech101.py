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


class Caltech101(Dataset):
    base_folder = '101_ObjectCategories'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        dataset = images.merge(image_class_labels, on='img_id')
        self.dataset = dataset.merge(train_test_split, on='img_id')

        if self.train:
            self.dataset = self.dataset[self.dataset.is_training_img == 1]
        else:
            self.dataset = self.dataset[self.dataset.is_training_img == 0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, idx
    
caltech_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ])
caltech_test_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def caltech_train_loader():
    return Caltech101('/srv/share/datasets/caltech101', train=True, transform=caltech_train_transform)


def caltech_test_loader():
    return Caltech101('/srv/share/datasets/caltech101', train=False, transform=caltech_test_transform)

def prepare_train_loaders(config):
    return {
        'full': caltech_train_loader()
    }

def prepare_test_loaders(config):
    return {
        'full': caltech_test_loader()
    }
