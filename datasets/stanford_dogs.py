import os
import pdb
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
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
from utils import create_heldout_split


class StanfordDogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False,
                 base_set=None):

        self.root_og = root
        self.name = 'stanford_dogs'
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform
        if download:
            self.download()

        split = self.load_split()
        self.images_folder = os.path.join(self.root, 'Images')
        if base_set is None:
            self.dataset = [(annotation+'.jpg', idx) for annotation, idx in split]
            self.targets = np.unique([idx for _, idx in split])
        else:
            self.dataset = base_set
            self.targets = np.unique([idx for _, idx in base_set])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self.dataset[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target_class#, index

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))
    
stanford_dogs_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

stanford_dogs_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def stanford_dogs_train_loader():
    return StanfordDogs('/srv/share/datasets/stanford_dogs', train=True, transform=stanford_dogs_train_transform)


def stanford_dogs_test_loader():
    return StanfordDogs('/srv/share/datasets/stanford_dogs', train=False, transform=stanford_dogs_test_transform)


def prepare_train_loaders(config):
    return {
        'full': torch.utils.data.DataLoader(
            StanfordDogs('/srv/share/datasets/stanford_dogs', train=True, transform=stanford_dogs_train_transform),
            batch_size=config['batch_size'],
            shuffle=config['shuffle_train'],
            num_workers=config['num_workers']
        )
            
    }

def prepare_test_loaders(config):
    loaders = {
        'full': torch.utils.data.DataLoader(
            StanfordDogs('/srv/share/datasets/stanford_dogs', train=False, transform=stanford_dogs_test_transform),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
    }
    if config.get('val_fraction', 0) > 0.:
        test_set = loaders['full'].dataset
        # val_set, test_set = train_test_split(test_set.dataset, test_size=config['val_fraction'])
        # val_set = OxfordPets('/srv/share/datasets/', train=False, transform=oxford_pets_test_transform, base_set=val_set)
        # test_set = OxfordPets('/srv/share/datasets/', train=False, transform=oxford_pets_test_transform, base_set=test_set)
        test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        loaders['heldout_val'] = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['heldout_test'] = torch.utils.data.DataLoader(
            test_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    return loaders
