
import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from timm.data import IMAGENET_INCEPTION_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


DEFAULT_CROP_RATIO = 224/256


class ImageNet1k:
    
    class DummyDataset:
        """ Dummy dataset to provide the length. """
        def __init__(self, len):
            self.len = len
        
        def __len__(self):
            return self.len

    def __init__(self,
                 path:str,
                 batch_size:int,
                 device:str,
                 res:int=224,
                 crop_ratio:float=DEFAULT_CROP_RATIO,
                 inception_norm:str=True,
                 num_workers:int=6):
        
        self.path = path
        self.batch_size = batch_size
        self.device = device

        self.res = res
        self.crop_ratio = crop_ratio
        self.inception_norm = inception_norm
        self.num_workers = num_workers


        self.train = self.create_loader('train_500_0.50_90.ffcv', val=False)
        self.test  = self.create_loader('val_500_0.50_90.ffcv', val=True)

        self.num_train = 1281167
        self.num_test  = 50000
        self.num_classes = 1000

        self.train.dataset = self.DummyDataset(self.num_train)
        self.test.dataset = self.DummyDataset(self.num_test)

    def create_loader(self, name, val):
        # Import here so that you don't need it to run the file
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Squeeze, RandomHorizontalFlip
        from ffcv.fields.decoders import IntDecoder, CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder

        if val:
            image_pipeline = [CenterCropRGBImageDecoder((self.res, self.res), ratio=self.crop_ratio)]
        else:
            image_pipeline = [
                RandomResizedCropRGBImageDecoder((self.res, self.res)),
                RandomHorizontalFlip()
            ]

        
        std  = IMAGENET_INCEPTION_STD  if self.inception_norm else IMAGENET_DEFAULT_STD
        mean = IMAGENET_INCEPTION_MEAN if self.inception_norm else IMAGENET_DEFAULT_MEAN
        mean, std = np.array(mean)*255, np.array(std)*255

        image_pipeline += [
            NormalizeImage(mean, std, np.float16),
            ToTensor(),
            ToDevice(torch.device(self.device), non_blocking=True),
            ToTorchImage(),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True),
        ]

        loader = Loader(
            os.path.join(self.path, name),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.SEQUENTIAL if val else OrderOption.QUASI_RANDOM,
            drop_last=False,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline,
            },
            distributed=False,
            seed=2023
        )

        return loader



def prepare_loaders(config):
    dataset = ImageNet1k(
        config['dir'], config['batch_size'], config['device'], config['res'],
        crop_ratio=config['crop_ratio'] if 'crop_ratio' in config else DEFAULT_CROP_RATIO,
        inception_norm=config['inception_norm'], num_workers=config['num_workers']
    )
    
    train_loader = { 'full': dataset.train }
    test_loader = { 'full': dataset.test }

    return train_loader, test_loader
