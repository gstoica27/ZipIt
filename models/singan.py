import math
import typing
import os
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
from PIL import Image
import pickle
import torchvision.transforms as transforms
import numpy as np
from .singan_utils import imresize
from utils import SpaceInterceptor



__all__ = ['g_multivanilla']

def initialize_model(model, scale=1.):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            continue





class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.batch_norm(self.conv(x)))
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(Vanilla, self).__init__()
        # parameters
        self.padding = (kernel_size // 2) * num_blocks

        # features
        blocks = [BasicBlock(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding)]
        for i in range(0, num_blocks - 2):
            f = max_features // pow(2, (i+1))
            blocks.append(BasicBlock(in_channels=max(min_features, f * 2), out_channels=max(min_features, f), kernel_size=kernel_size, padding=padding))
        self.features = nn.Sequential(*blocks)
        
        # classifier
        self.features_to_image = nn.Sequential(
            nn.Conv2d(in_channels=max(f, min_features), out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.Tanh())
        
        # initialize weights
        initialize_model(self)
        
        # Added to be able to stop at output of Scaling Block
        self.intercept1 = SpaceInterceptor(dim=in_channels)

    def forward(self, z, x):
        # try:
        u = F.pad(z, [self.padding, self.padding, self.padding, self.padding])
        v = self.features(u)
        z_ = self.features_to_image(v)
        y = x + z_
        o = self.intercept1(y, kind='conv')
        return o.contiguous()

class MultiVanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(MultiVanilla, self).__init__()
        # parameters
        self.in_channels = in_channels
        self.max_features = max_features
        self.min_features = min_features
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.scale = 0
        self.key = 's0'
        self.scale_factor = 0

        # current
        self.curr = Vanilla(in_channels, max_features, min_features, num_blocks, kernel_size, padding)
        self.prev = nn.Module()
        self.idx = 3

    def add_scale(self, device):
        self.scale += 1

        # previous
        self.prev.add_module(self.key, deepcopy(self.curr))
        self._reset_grad(self.prev, False)
        self.key = 's{}'.format(self.scale)

        # current
        max_features = min(self.max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(self.min_features * pow(2, math.floor(self.scale / 4)), 128)
        if math.floor(self.scale / 4) != math.floor((self.scale - 1) / 4):
            self.curr = Vanilla(self.in_channels, max_features, min_features, self.num_blocks, self.kernel_size, self.padding).to(device)

    def _compute_previous(self, reals, amps, noises=None):
        # parameters
        keys = list(reals.keys())
        y = torch.zeros_like(reals[keys[0]])
            
        # loop over scales
        for key, single_scale in self.prev.named_children():
            # print(key)
            next_key = keys[keys.index(key) + 1]
            # fixed z
            if noises:
                z = y + amps[key].view(-1, 1, 1, 1) * noises[key]
            # random noise
            else:
                n = self._generate_noise(reals[key], repeat=(key == 's0'))
                z = y + amps[key].view(-1, 1, 1, 1) * n
            y = single_scale(z, y)
            y = imresize(y, 1. / self.scale_factor)
            y = y[:, :, 0:reals[next_key].size(2), 0:reals[next_key].size(3)]
            
        return y

    def forward(self, reals, amps, noises=None, seed=None):
        # compute prevous layers
        with torch.no_grad():
            y = self._compute_previous(reals, amps, noises).detach()
            
        # fixed noise
        if noises:
            z = y + amps[self.key].view(-1, 1, 1, 1) * noises[self.key]
        # random noise
        else:
            n = self._generate_noise(reals[self.key], repeat=(not self.scale))
            z = y + amps[self.key].view(-1, 1, 1, 1) * n

        o = self.curr(z.detach(), y.detach()) 
        return o

    def _download_pickle(self, tensor):
        directory = './checkpoints/singan/noises/'
        idx = len(os.listdir(directory))
        pickle.dump(tensor, open(os.path.join(directory, f'model_{idx}.pkl'), 'wb'))
        
    def _load_pickle(self):
        directory = './checkpoints/singan/noises/'
        idx = self.idx
        return pickle.load(open(os.path.join(directory, f'model_{idx}.pkl'), 'rb'))
    
    def _generate_noise(self, tensor_like, repeat=False):
        if not repeat:
            # noise = torch.randn(tensor_like.size()).to(tensor_like.device)
            noise = torch.tensor(np.random.randn(*list(tensor_like.size())), device=tensor_like.device)
        else:
            noise = torch.tensor(
                np.random.randn(
                    tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)
                )
            )
            # noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
            noise = noise.repeat((1, 3, 1, 1)).to(tensor_like.device)
        # self._download_pickle(noise)
        return noise.to(torch.float32)

    def _reset_grad(self, model, require_grad=False):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def train(self, mode=True):
        self.training = mode
        # train
        for module in self.curr.children():
            module.train(mode)
        # eval
        for module in self.prev.children():
            module.train(False)
        return self

    def eval(self):
        self.train(False)

def g_multivanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    
    return MultiVanilla(**config)









class Sampler(nn.Module):

    def _init_eval(self, base_path, data_path, device='cuda'):
        # paramaters 
        self.scale = 0
        self.step = 0
        self.batch_size = 1

        # path_to_img = os.path.join(data_dir, 'img.png')
        path_to_img = data_path
        path_to_model = os.path.join(base_path, 'g_multivanilla.pt')
        path_to_amps = os.path.join(base_path, 'amps.pt')

        image = Image.open(path_to_img).convert('RGB')
        image = transforms.ToTensor()(image).unsqueeze(dim=0)
        image = (image - 0.5) * 2
        image = image.to(device)

        min_size = 25
        max_size = 250

        scale_factor = 0.75
        scale_factor_init = 0.75

        num_scales = math.ceil((math.log(math.pow(min_size / (min(image.size(2), image.size(3))), 1), scale_factor_init))) + 1
        scale_to_stop = math.ceil(math.log(min([max_size, max([image.size(2), image.size(3)])]) / max([image.size(2), image.size(3)]), scale_factor_init))
        stop_scale = num_scales - scale_to_stop

        scale_one = min(max_size / max([image.size(2), image.size(3)]), 1)
        image_resized = imresize(image, scale_one)

        scale_factor = math.pow(min_size/(min(image_resized.size(2), image_resized.size(3))), 1 / (stop_scale))
        scale_to_stop = math.ceil(math.log(min([max_size, max([image_resized.size(2), image_resized.size(3)])]) / max([image_resized.size(2), image_resized.size(3)]), scale_factor_init))
        stop_scale = num_scales - scale_to_stop
        
        reals = {}
        for i in range(stop_scale + 1):
            s = math.pow(scale_factor, stop_scale - i)
            reals.update({'s{}'.format(i): imresize(image_resized.clone().detach(), s).squeeze(dim=0)})
        self.reals = reals

        max_features = 32
        min_features = 32
        num_blocks = 5
        kernel_size = 3
        padding = 0

        # number of features
        max_features = min(max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(min_features * pow(2, math.floor(self.scale / 4)), 128)

        # config
        model_config = {'max_features': max_features, 'min_features': min_features, 'num_blocks': num_blocks, 'kernel_size': kernel_size, 'padding': padding}        

        # init first scale
        g_model = g_multivanilla
        self.g_model = g_model(**model_config)
        self.g_model.scale_factor = scale_factor

        # add scales
        for self.scale in range(1, stop_scale + 1):
            self.g_model.add_scale('cpu')

        # load model
        self.g_model.load_state_dict(torch.load(path_to_model, map_location='cpu'), strict=False)
        self.amps = torch.nn.ParameterDict(torch.load(path_to_amps, map_location='cpu'))

        # cuda
        self.g_model = self.g_model.to(device)
        self.amps.to(device)
        # for key in self.amps.keys():
        #     self.amps[key] = self.amps[key].to(device)
    

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        data_reals = self.reals
        reals = {}
        amps = self.amps

        # set reals
        for key in data_reals.keys():
           reals.update({key: data_reals[key].clone().unsqueeze(dim=0).repeat(x.shape[0], 1, 1, 1)}) 

        seed = int(x.abs().mean() * 12591250125 + 1249612)

        # evaluation
        with torch.no_grad():
            generated_sampled = self.g_model(reals, amps, seed=seed)

        return (generated_sampled + 1.) / 2.
        # # save image
        # self._save_image(generated_sampled, 's{}_sampled.png'.format(self.step))
    
    def tensor_to_image(self, x):
        ndarr = x.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        return image

    def _save_image(self, image, save_path=None):
        image = self.tensor_to_image(image.data.cpu()[0])
        # x = image
        # ndarr = x.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        import matplotlib.pyplot as plt
        
        # fig, ax = plt.subplots(3, 2)
        # ax[0, 0].imshow(ndarr[:, :, [0, 1, 2]])
        # ax[0, 1].imshow(ndarr[:, :, [0, 2, 1]])
        
        # ax[1, 0].imshow(ndarr[:, :, [1, 0, 2]])
        # ax[1, 1].imshow(ndarr[:, :, [1, 2, 0]])
        
        # ax[2, 0].imshow(ndarr[:, :, [2, 0, 1]])
        # ax[2, 1].imshow(ndarr[:, :, [2, 1, 0]])
        # plt.show()
        
        plt.imshow(image)
        plt.show()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path))
            os.save_image_grid(image.data.cpu(), save_path)



if __name__ == '__main__':
    model = Sampler()
    model._init_eval('./checkpoints/singan/birds/', device=0)
    
    model._save_image(model(0))
