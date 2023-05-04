import torch
import torchvision
import numpy as np

from .imagenet import ImageNet1k
from .nabird import NABird
from .cub import CUB2011
from .oxford_pets import OxfordPets
from .stanford_dogs import StanfordDogs
from .caltech101 import Caltech101

cifar50 = {
    'dir': './data/cifar-100-python', 
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 50,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 2,
}

cifar5 = {
    'dir': './data/cifar-10-python',
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}

cifar10 = {
    'dir': './data/cifar-10-python',
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}

imagenet1k = {
    'dir': './data/ffcv/',
    'num_classes': 1000,
    'wrapper': ImageNet1k,
    'batch_size': 16,
    'res': 224,
    'inception_norm': True,
    'shuffle_test': False,
    'type': 'imagenet',
    'num_workers': 8,
}

nabird = {
    'wrapper': NABird,
    'batch_size': 8,
    'res': 224,
    'type': 'nabird',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/nabirds'
}

cub = {
    'wrapper': CUB2011,
    'batch_size': 8,
    'res': 224,
    'type': 'cub',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/cub200'
}

caltech101 = {
    'wrapper': Caltech101,
    'batch_size': 8,
    'res': 224,
    'type': 'caltech101',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/caltech101'
}

stanford_dogs = {
    'wrapper': StanfordDogs,
    'batch_size': 8,
    'res': 224,
    'type': 'stanford_dogs',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
    'dir': './data/stanford_dogs'
}

oxford_pets = {
    'wrapper': OxfordPets,
    'batch_size': 8,
    'res': 224,
    'type': 'oxford_pets',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/oxford_pets'
}