from collections import defaultdict
from collections.abc import MutableMapping
from typing import Sequence, Callable
import os
import math
import pdb
import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm.auto import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from fvcore.nn.flop_count import flop_count
from inspect import getmembers, isfunction
from metric_calculators import get_metric_fns
import torch.nn.functional as F
import clip
import einops
import torch
import scipy
import random
import string


CONCEPT_TASKS  = list(string.ascii_uppercase)

##########################################################################################################################
######################################################### CLASSES ########################################################
##########################################################################################################################

class SpaceInterceptor(nn.Module):
    '''
    This module is meant to intercept computational flows between any given two layers. 
    Inserting the module between two layers allows us to compute a merge/unmerge on each 
    layer separately, rather than a single merge/unmerge for both. This is most useful for
    controlling the transformations learned over residual connections. E.g., if we have a 
    case where we combine several residuals together, we can instead place this on each 
    branch before their connection, allowing us to learn distinct merge/unmerges on each
    branch, and 1 merge/unmerge on the connection, rather than 1 merge/unmerge for everything.
    Thus, it allows for (hopefully) more specificity.
    
    All it requires is a dimension parameter (the size of the feature dimension).
    
    It contains only 1 weight, which begins as the identity, and will be transformed according to
    the unmerge/merge that will be applied over it. For all intents and purposes, this is treated
    as a linear layer, with not bias! 
    '''
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim))
    
    def forward(self, input, kind='linear'):
        if kind == 'conv':
            input = input.permute(0, 2,3, 1)
        
        output = input @ self.weight.T
        
        if kind == 'conv':
            output = output.permute(0, 3, 1, 2)
        
        return output
    

class SpoofModel(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""
    def __init__(self, models):
        super().__init__()
        self.models = models
        
    def forward(self, x):
        """Call all models returning list of their outputs."""
        return [model(x) for model in self.models]
    
    def parameters(self):
        """Return list of parameters from first model."""
        return self.models[0].parameters()


class DummyDataset:
    """ Dummy dataset to provide the length. """
    def __init__(self, len):
        self.len = len
    
    def __len__(self):
        return self.len


class FractionalDataloader:
    def __init__(self, dataloader, fraction, seed=None):
        self.dataloader_numel = len(dataloader.dataset)
        self.numel = int(fraction * self.dataloader_numel)

        self.batch_size = self.dataloader_numel / len(dataloader)
        self.num_batches = int(math.ceil(self.numel / self.batch_size))
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.seed = seed
    
    def __iter__(self):
        cur_elems = 0
        if self.seed is not None:
            self.dataloader.dataset.set_seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
        it = iter(self.dataloader)
        while cur_elems < self.numel:
            try:
                x, y = next(it)
                cur_elems += x.shape[0]
                yield x, y
            except StopIteration:
                it = iter(self.dataloader)
                
        
    def __len__(self):
        return self.num_batches


class SpoofLoader(object):
    def __init__(self, *dataloaders):
        """Join multiple dataloaders together."""
        super().__init__()
        self.dataloaders = dataloaders
        self.dataset = DummyDataset(min(len(dataloader.dataset) for dataloader in dataloaders))
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterate over all dataloaders getting the images and labels in a concatenated form."""
        num_loaders = len(self.dataloaders)
        for _ in zip(*self.dataloaders):
            images = []
            labels = []
            for loader_images, loader_labels in _:
                images.append(loader_images)
                labels.append(loader_labels)
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)
            yield images, labels


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, by_loss=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = -np.inf

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


##########################################################################################################################
################################################## TRAIN/EVAL FUNCTIONS ##################################################
##########################################################################################################################

def evaluate_cliphead_alltasks(model, loader, label_encodings_list, splits, num_classes):
    """Evaluate a cliphead on all tasks. Return acc_overall, acc_avg, perclass_acc
    
    Args:
        model: cliphead model
        loader: dataloader to evaluate on
        label_encodings_list: list of clip label encodings
        splits: list of classes for each task definition
        num_classes: number of classes in the dataset
    Returns:
        acc_overall: overall accuracy
        acc_avg: average accuracy
        perclass_acc: accuracy per class
    """

    model.eval()
    device = get_device(model)
    correct = 0
    total = 0
    
    all_splits = torch.tensor(sum(splits, [])).to(device)
    splits = [list(split) for split in splits]
    totals = [0] * num_classes
    corrects = [0] * num_classes
    
    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i

    task_map = [task_map[_cls] if _cls in task_map else -1 for _cls in range(num_classes)]
    task_map = torch.tensor(task_map).to(device)
    splits = torch.tensor(splits).to(device)
    label_encodings = torch.stack(label_encodings_list, dim=0).to(device) # [S,C,D]
    multihead = False
    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating model on CLIP class encodings'):
            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue
            encodings = model(inputs.to(device))
            task_idx = task_map[labels]
            task_splits = splits[task_idx, :]

            if isinstance(encodings, list):
                multihead = True
                encodings = torch.stack(encodings, dim=0)
            encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            if len(encodings.shape) == 3:
                all_logits = torch.bmm(encodings, label_encodings.transpose(-1, -2)).transpose(1, 0)
            else:
                all_logits = torch.einsum('be,sec->bsc', encodings, label_encodings.transpose(-1, -2))
            
            if multihead:
                all_logits = F.softmax(all_logits * 100, dim=-1)
            task_preds = all_logits[range(batch_size), task_idx, :].argmax(-1)
            task_preds = task_splits.gather(dim=-1, index=task_preds[:, None])[:, 0]
            all_preds = all_logits.flatten(1)[:, all_splits.argsort()].argmax(-1)
            
            for gt, task_p, all_p in zip(labels, task_preds, all_preds):
                totals[gt] += 1
                if gt == task_p:
                    corrects[gt] += 1
                if gt == all_p:
                    correct += 1
                total += 1

        split_accs = [0] * len(splits)
        for i, split in enumerate(splits):
            split_total = 0
            for _cls in split:
                split_accs[i] += corrects[_cls]
                split_total += totals[_cls]
            split_accs[i] /= max(split_total, 1e-4)

        return correct / total, sum(split_accs) / len(split_accs), split_accs
            

# evaluates accuracy
def evaluate_cliphead(
    model, loader, class_vectors, remap_class_idxs=None, 
    return_confusion=False, task_info=None, return_loss=False):
    """Evaluate a model with a cliphead on a dataset.
    
    Args:
        model: cliphead model
        loader: dataloader to evaluate on
        class_vectors: clip label encodings
        remap_class_idxs: array or mapping from true class labels to those expected given the task
        return_confusion: whether to return the confusion matrix
        task_info: dictionary containing task information
        return_loss: whether to return the loss
    Returns:
        accuracy: accuracy of the model
        confusion: confusion matrix
        loss: loss of the model

    """
    model.eval()
    correct = 0
    total = 0
    
    totals = np.array([0] * class_vectors.shape[0])
    corrects = np.array([0] * class_vectors.shape[0])

    device = get_device(model)
    losses = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating CLIP head model'):
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            
            if task_info is not None:
                task_map = task_info['task_map']
                data_label_task = task_map[labels].to(device)
                task_features = torch.stack(task_info['task_features'], dim=0).transpose(-1, -2)[data_label_task]
                outputs = torch.einsum('ij,ijk->ik', normed_encodings, task_features)
                remap_class_idxs = task_info['remap_class_idxs']
            else:
                outputs = normed_encodings @ class_vectors.T
            pred = outputs.argmax(dim=1)
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels]
            else:
                remapped_labels = labels
            loss = loss_fn(outputs, remapped_labels.to(device))
            losses += [loss.item()]

            for gt, p in zip(labels, pred):
                if remap_class_idxs is not None:
                    idx = gt
                    gt = remap_class_idxs[gt]
                else:
                    idx = gt
                
                is_correct = (gt == p).item()
                correct += is_correct
                
                 
                if return_confusion:
                    totals[idx] += 1
                    corrects[idx] += is_correct
                    
            total += inputs.shape[0]
    
    overall_loss = np.mean(losses)

    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / a[1], zip(corrects, totals)))
    else:
        if return_loss:
            return correct / total, overall_loss
        return correct / total
    

def train_cliphead(model, train_loader, test_loader, class_vectors, remap_class_idxs=None, epochs=200):
    """Train a cliphead model.
    
    Args:
        model: cliphead model
        train_loader: dataloader to train on
        test_loader: dataloader to test on
        class_vectors: clip label encodings
        remap_class_idxs: array or mapping from true class labels to those expected given the task
        epochs: number of epochs to train for
    Returns:
        model: trained cliphead model
        train_acc: training accuracy
    """
    optimizer = SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)
    ne_iters = len(train_loader)
    lr_schedule = np.interp(np.arange(1+epochs*ne_iters), [0, 5*ne_iters, epochs*ne_iters], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    device = get_device(model)
    
    losses = []
    acc = 0.
    pbar = tqdm(range(epochs), desc=f'finetuning, prev acc: {acc}: ')
    for _ in pbar:
        model = model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                encodings = model(inputs.to(device))
                normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                logits = (100.0 * normed_encodings @ class_vectors.T)
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)
                # pdb.set_trace()
                loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            
        acc = evaluate_cliphead(model, test_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs)
        pbar.set_description(f'finetuning, prev acc: {acc}: ')
        print(f'Epoch {_}, Acc: {acc}')
    acc = evaluate_cliphead(model, test_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs)
    return model, acc


def evaluate_logits_alltasks(model, loader, splits, num_classes):
    model.eval()
    correct = 0
    total = 0
    
    splits = [list(split) for split in splits]

    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    all_splits = torch.tensor(sum(splits, [])).to(device)

    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i
    
    task_map = [task_map[_cls] if _cls in task_map else -1 for _cls in range(num_classes)]
    task_map = torch.tensor(task_map).to(device)

    splits = torch.tensor(splits).to(device)

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating multihead head model'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector, :, :, :], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]
            outputs = model(inputs)

            if isinstance(outputs, list):
                # Filter out predictions on classes not trained
                for i in range(len(outputs)):
                    exclude_labels = torch.tensor([l for l in all_splits.cpu().numpy() if l not in splits[i].cpu().numpy()], device=all_splits.device)
                    outputs[i][:, exclude_labels] = -torch.inf
                outputs = torch.stack(outputs, dim=1)
                outputs2 = outputs.softmax(dim=-1).to(outputs.dtype).max(dim=-2)[0]
                outputs2[:, all_splits] += 2
                outputs = outputs[range(batch_size), task_idx, :]
            else:
                outputs2 = outputs.clone()
                for i in range(splits.shape[0]):
                    outputs2[:, splits[i]] = torch.softmax(outputs2[:, splits[i]], dim=-1).to(outputs.dtype) + 2
            outputs2 = outputs2.argmax(dim=-1)

            task_splits = splits[task_idx, :]
            outputs = outputs.gather(dim=-1, index=task_splits).argmax(dim=-1)
            outputs = task_splits.gather(dim=-1, index=outputs[:, None])[:, 0]

            for gt, p, p2 in zip(labels, outputs, outputs2):
                totals[gt] += 1
                
                if gt == p:
                    corrects[gt] += 1
                if gt == p2:
                    correct += 1
                
                total += 1

    split_accs = [0] * len(splits)

    for i, split in enumerate(splits):
        split_total = 0
        for _cls in split:
            split_accs[i] += corrects[_cls]
            split_total += totals[_cls]
        split_accs[i] /= max(split_total, 1e-4)
                
    return correct / total, sum(split_accs) / len(split_accs), split_accs


def evaluate_logits_i(model, loader, head_index, num_classes, return_confusion=False):
    model.eval()
    correct = 0
    total = 0

    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating multihead head model'):
            inputs, labels = inputs.to(device), labels.to(device)

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue
            
            logits = model(inputs)[head_index].argmax(-1)
            for gt, p in zip(labels, logits):
                try:
                    totals[gt] += 1
                except:
                    pdb.set_trace()
                
                if gt == p:
                    correct += 1
                    corrects[gt] += 1
                
                total += 1
    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / max(a[1], 1e-4), zip(corrects, totals)))
    else:
        return correct / total
    

def train_logits(model, train_loader, test_loader, epochs=200, remap_class_idxs=None):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.008)
    ne_iters = len(train_loader)
    # scheduler = lr_scheduler.LinearLR(optimizer, min_lr=.0000001, verbose=True, factor=np.sqrt(.1), cooldown=0.)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-7, total_iters=ne_iters)
    early_stopper = EarlyStopper(patience=epochs, min_delta=.0001)
    
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction='mean')
    device = get_device(model)
    losses = []
    acc = 0.
    best_acc = 0.
    best_epoch = 0
    best_sd = None
    pbar = tqdm(range(epochs), desc=f'Training, prev acc: {acc}: ')
    for epoch in pbar:
        model = model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(inputs.to(device))
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)
                    
                loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss)
            losses.append(loss.item())

        acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
        if acc > best_acc:
            best_sd = model.state_dict()
            best_acc = acc
            best_epoch = epoch
            
        if early_stopper.early_stop(acc):
            print(f'Stopping at Epoch: {epoch}. Best Accuracy {best_acc}, Achieved at Epoch {best_epoch}')
            break
        pbar.set_description(f'Training, prev acc: {acc}: ')
    model.load_state_dict(best_sd)
    acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
    print('Acc at Best Model: {}'.format(acc))
    return model, best_acc


def evaluate_logits(model, test_loader, return_confusion=False, use_flip_aug=False, remap_class_idxs=None, class_idxs=None, eval_mask=None):
    model.eval()
    correct = 0
    total = 0
    totals = defaultdict(lambda: 0)
    corrects = defaultdict(lambda: 0)
    loss_fn = CrossEntropyLoss()
    device = next(iter(model.parameters())).device
    total_loss = 0
    total_iter = len(test_loader)
    with torch.no_grad(), autocast():
        for _ in tqdm(test_loader, 'Evaluating classification model'):
            inputs, labels = _
            inputs = inputs.to(device)
            outputs = model(inputs)
            if use_flip_aug:
                outputs += model(torch.flip(inputs, (3,)))
            
            if eval_mask is not None:
                outputs[:, eval_mask == 0] = -torch.inf
            
            pred = outputs.argmax(dim=-1)
            # pdb.set_trace()
            total += pred.shape[0]
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels].to(device)
            else:
                remapped_labels = labels.to(device)
            # pdb.set_trace()
            loss = loss_fn(outputs, remapped_labels)
            total_loss += loss
            for gt, p in zip(remapped_labels, pred):
                gt, p = gt.item(), p.item()
                totals[gt] += 1

                if gt == p:
                    correct += 1
                    corrects[gt] += 1

    num_classes = max(totals)+1
    totals = [totals[i] for i in range(num_classes)]
    corrects = [corrects[i] for i in range(num_classes)]
    total_loss = total_loss / total_iter
    
    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / a[1], zip(corrects, totals)))
    else:
        return correct / total


def evaluate_model(eval_type, model, config, **opt_kwargs):
    """ Evaluate methods on arbitrary experiment kinds. """
    if opt_kwargs.get("opt_dataloader", None) is not None:
        loader = opt_kwargs["opt_dataloader"]
        num_classes = opt_kwargs["opt_classes"]
    else:
        loader = config['data']['test']['full']
        num_classes = len(config['data']['test']['class_names'])
        
    if eval_type == 'logits':    
        acc_overall, acc_avg, perclass_acc = evaluate_logits_alltasks(
            model, loader, 
            splits=config['dataset']['class_splits'], 
            num_classes=num_classes
        )
        
    elif eval_type == 'clip':
        clip_features = load_clip_features(config['data']['test']['class_names'], get_device(model))
        class_vectors = [clip_features[split] for split in config['data']['train']['class_splits']]

        acc_overall, acc_avg, perclass_acc = evaluate_cliphead_alltasks(
            model, 
            loader, 
            class_vectors, config['data']['train']['class_splits'], 
            num_classes=num_classes
        )
    else:
        raise ValueError(f'Invalid eval_type: {eval_type}! Must be one of [logits, clip].')

    results = {'Joint': acc_overall, 'Per Task Avg': acc_avg}
    for task_idx, task_acc in enumerate(perclass_acc):
        results[f'Task {CONCEPT_TASKS[task_idx]}'] = task_acc

    return results


##########################################################################################################################
############################################### EXPERIMENT CONFIG CREATION ###############################################
##########################################################################################################################

def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]
    
    dataset_name = config['name']
    
    import datasets.configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device

    if data_config['type'] == 'cifar':
        from datasets.cifar import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'imagenet':
        from datasets.imagenet import prepare_loaders
        train_loaders, test_loaders = prepare_loaders(data_config)
    elif data_config['type'] == 'nabird':
        from datasets.nabird import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'cub':
        from datasets.cub import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'oxford_pets':
        from datasets.oxford_pets import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'stanford_dogs':
        from datasets.stanford_dogs import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    else:
        raise NotImplementedError(config['type'])
    
    if 'train_fraction' in data_config:
        for k, v in dict(train_loaders.items()).items():
            if k == 'splits':
                train_loaders[k] = [FractionalDataloader(x, data_config['train_fraction']) for x in v]
            elif not isinstance(v, list) and not isinstance(v, torch.Tensor):
                train_loaders[k] = FractionalDataloader(v, data_config['train_fraction'])

    return {
        'train': train_loaders,
        'test': test_loaders
    }


def prepare_resnets(config, device):
    """ Load all pretrained resnet models in config. """
    bases = []
    name = config['name']

    if 'x' in name:
        width = int(name.split('x')[-1])
        name = name.split('x')[0]
    else:
        width = 1

    if 'resnet20' in name:
        from models.resnets import resnet20 as wrapper_w
        wrapper = lambda num_classes: wrapper_w(width, num_classes)
    elif 'resnet50' in name:
        from torchvision.models import resnet50 as wrapper
    elif 'resnet18' in name:
        from torchvision.models import resnet18 as wrapper
    else:
        raise NotImplementedError(config['name'])
    
    output_dim = config['output_dim']
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        base_sd = torch.load(base_path, map_location=torch.device(device))
        
        # Remove module for dataparallel
        for k in list(base_sd.keys()):
            if k.startswith('module.'):
                base_sd[k.replace('module.', '')] = base_sd[k]
                del base_sd[k]

        base_model = wrapper(num_classes=output_dim).to(device)
        base_model.load_state_dict(base_sd)
        bases.append(base_model)
    new_model = wrapper(num_classes=output_dim).to(device)
    return {
        'bases': bases,
        'new': new_model # this will be the merged model
    }


def prepare_singan(config, device):
    """ Load all pretrained singan models in config. """
    from models.singan import Sampler

    bases = []
    for i, base_path in tqdm(enumerate(config['bases']), desc="Preparing Models"):
        base_model = Sampler()
        base_model._init_eval(base_path, config['data_paths'][i], device)
        bases.append(base_model)
        
    new_model = Sampler()
    new_model._init_eval(base_path, config['data_paths'][i], device) # this will be merged model.

    return { 'bases': bases, 'new': new_model }

def prepare_vgg(config, device):
    """ Load all pretrained vgg models in config. """
    if 'vgg11' in config['name']:
        from models.vgg import vgg11 as wrapper_w
    elif 'vgg16' in config['name']:
        from models.vgg import vgg16 as wrapper_w
    else:
        raise ModuleNotFoundError(config['name'])
    bases = []
    name = config['name']
    if '_w' in name:
        width = int(name.split('_w')[-1])
        name = name.split('_w')[0]
    else:
        width = 1
    wrapper = lambda num_classes: wrapper_w(width, num_classes)
    output_dim = config['output_dim']
    
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        base_sd = torch.load(base_path, map_location=torch.device(device))
        
        base_model = wrapper(num_classes=output_dim).to(device)
        base_model.load_state_dict(base_sd)
        bases.append(base_model)
    new_model = wrapper(num_classes=output_dim).to(device)
    return {
        'bases': bases,
        'new': new_model # this will be merged model
    }

def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('resnet'):
        return prepare_resnets(config, device)
    elif config['name'].startswith('singan'):
        return prepare_singan(config, device)
    elif config['name'].startswith('vgg'):
        return prepare_vgg(config, device)
    else:
        raise NotImplementedError(config['name'])


def prepare_graph(config):
    """ Get graph class of experiment models in config. """
    if config['name'].startswith('resnet'):
        model_name = config['name'].split('x')[0]
        import graphs.resnet_graph as graph_module
        graph = getattr(graph_module, model_name)
    elif config['name'].startswith('singan'):
        from graphs.singan_graph import SinGANGraph as graph
    elif config['name'].startswith('vgg'):
        model_name = config['name'].split('_w')[0]
        import graphs.vgg_graph as graph_module
        graph = getattr(graph_module, model_name)
    else:
        raise NotImplementedError(config['name'])
    return graph


def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]


def prepare_experiment_config(config):
    """ Load all functions/classes/models requested in config to experiment config dict. """
    data = prepare_data(config['dataset'], device=config['device'])
    if config['eval_type'] == 'logits':
        config['model']['output_dim'] = len(data['test']['class_names'])
    else:
        config['model']['output_dim'] = 512
    new_config = {
        'graph': prepare_graph(config['model']),
        'data': data,
        'models': prepare_models(config['model'], device=config['device']),
        'merging_fn': get_merging_fn(config['merging_fn']),
        'metric_fns': get_metric_fns(config['merging_metrics']),
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config

def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    out = deepcopy(getattr(__import__('configs.' + name), name).config)
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


##########################################################################################################################
#################################################### HELPER FUNCTIONS ####################################################
##########################################################################################################################

def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
        
def convert_dict_to_tuple(d):
    """Convert a dictionary to a tuple of key-value pairs."""
    return tuple(list(d.items()))


def flatten_nested_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary. {a: {b: 1}} -> {a_b: 1}"""
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def write_to_csv(results, csv_file):
    """Write results to a csv file."""
    if not os.path.exists(csv_file):
        # Create dir if necessary
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        keys = list(results.keys())
        # Remove '_' and Capitalize first letter of every word
        keys = [str(key).replace('_', ' ').title() for key in keys]
        names = ','.join(keys)
        with open(csv_file, 'a') as f:
            f.write(f"{names}\n")
    
    csv_line = ','.join([str(i) for i in results.values()])
    with open(csv_file, 'a') as f:
        f.write(f"{csv_line}\n")
        

def vector_gather(vectors, indices):
    """
    from: https://gist.github.com/EricCousineau-TRI/cc2dc27c7413ea8e5b4fd9675050b1c0
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


# use the train loader with data augmentation as this gives better results
# taken from https://github.com/KellerJordan/REPAIR
def reset_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            if reset:
                m.momentum = None # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad(), autocast():
        for images, _ in tqdm(loader, desc='Resetting batch norm'):
            _ = model(images.to(device))
    return model

def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


def load_clip_features(class_names, device):
    """Create CLIP target labels for class names. Return a normalized tensor of shape (num_classes, 512)."""
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    model, preprocess = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def find_pairs(str_splits):
    pairs = []
    for i, str_split_i in enumerate(str_splits):
        try:
            split_i = set([int(k) for k in str_split_i.split('_')])
        except:
            continue
        for str_split_j in str_splits[i+1:]:
            try:
                split_j = set([int(k) for k in str_split_j.split('_')])
            except:
                continue
            if len(split_i.intersection(split_j)) == 0:
                pairs.append((str_split_i, str_split_j))
    return pairs


def find_runable_pairs(model_dir, model_name, skip_pair_idxs=[]):
    run_pairs = []
    valid_pairs = [pair for pair in find_pairs(os.listdir(model_dir)) if is_valid_pair(model_dir, pair, model_name)]
    for idx, pair in enumerate(valid_pairs):
        if idx in skip_pair_idxs:
            continue
        run_pairs += [pair]
    return run_pairs


def split_str_to_ints(split):
    return [int(i) for i in split.split('_')]


def is_valid_pair(model_dir, pair, model_type):
    paths = os.listdir(os.path.join(model_dir, pair[0]))
    flag = True
    for path in paths:
        if f'{model_type}_v0.pth.tar' not in path:
            flag = False
    return flag


def create_heldout_split(dataset, fraction):
    root = dataset.root_og
    val_set, test_set = train_test_split(dataset.dataset, test_size=fraction)
    val_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=val_set)
    test_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=test_set)
    return val_set, test_set
    

def save_model(model, save_path):
    sd = model.state_dict()
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path, model_device='cuda'):
    sd = torch.load(save_path, map_location=torch.device(model_device))
    model.load_state_dict(sd)
    return model
    
def read_yaml(path):
    with open(path) as handle:
        try:
            config = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            print(error)
    return config


def inject_pair(config, pair, ignore_bases=False):
    model_name = config['model']['name']
    config['dataset']['class_splits'] = [split_str_to_ints(split) for split in pair]
    if not ignore_bases:
        config['model']['bases'] = [os.path.join(config['model']['dir'], split, f'{model_name}_v0.pth.tar') for split in pair]
    return config

    
def get_contour_grid(models: Sequence, eval_fn: Callable, n_points_row: int = 10, n_points_col: int= 10, basis: Sequence = None, scale_row: float = None, scale_col: float = None):
  """
  Function taking in a sequence of models and creating a grid 
  representing interpolations between the models, along with 
  the locations of the original models, in their given order 
  on the returned grid.
  Heavily inspired by Stanislav Fort's code at https://github.com/stanislavfort/dissect-git-re-basin,
  exteded to work in the n-models case. Expects the n models to be closely projectable to a 2D space.

    Args:
        models: a sequence of model state dicts, need 3 models 
        minimum if no basis is given
        eval_fn: a function taking in a model state dict and
            returning the evaluation of the model at that point
        n_points: the number of points to use for each dimension
            of the grid. The total number of points will be n_points**2
        basis: a sequence of unit vectors, each of the same length as
            the flattened model, representing the basis vectors to use
            for the interpolation. If not given, the first two principal 
            axis of the list of models will be used. 
        scale_row: a float representing the scale of the basis vectors in row direction.
        scale_col: a float representing the scale of the basis vectors in col direction.
    Returns:
        a dictionary of the form
        {
            "grid": a numpy array of shape (n_points, n_points)
            "model_locations": list of tuples representing the
                locations of the models on the grid
            "tick_cols": the ticks along the columns
            "tick_rows": the ticks along the rows
            "basis": the basis vectors used for the interpolation
            "scale": the scale of the basis vectors
        }
  
  """
  # reconstructing parameter dictionaries from flat vectors
  def reconstruct(vector, example_flat_model, keys):
    i = 0
    output = dict()
    for key in keys:
      shape_now = example_flat_model[key].shape
      size_now = int(np.prod(shape_now))
      data_now = vector[i:i+size_now].reshape(shape_now)
      output[key] = data_now
      i = i + size_now
    return output
  keys = models[0].keys()
  models_flattened = [torch.concat([model[key].reshape([-1]) for key in keys],axis=0) for model in models]
  average_model = torch.mean(torch.stack(models_flattened, axis=1), axis=1)
  models_flattened = [model - average_model for model in models_flattened]
  if basis is None:
    # get the first two principal components with SVD
    U, S, V = torch.linalg.svd(torch.stack(models_flattened, axis=1), full_matrices=False)
    basis = [U[:,0], U[:,1]]
  def project_basis(model_vector):
    return torch.Tensor([torch.sum(model_vector*basis[0]), torch.sum(model_vector*basis[1])])
  def project_grid(base_vector, model_vector, scale_row, scale_col):
    return (project_basis(model_vector) - project_basis(base_vector)) / torch.Tensor([scale_row, scale_col])
  def unprojection(row, col):
    return row*basis[0] + col*basis[1] + average_model
  max_col = max_row = 0
  min_col = min_row = float("inf")
  for i, model in enumerate(models_flattened):
    row, col = project_basis(model)
    if col < min_col:
      min_col = col
    if col > max_col:
      max_col = col
    if row < min_row:
      min_row = row
    if row > max_row:
      max_row = row
  if scale_row is None:
    scale_row = max_row - min_row
    scale_col = max_col - min_col
  # get the grid
  t1s = torch.linspace(-0.5,1.5,n_points_row)
  t2s = torch.linspace(-0.5,1.5,n_points_col)
  grid_acc = torch.zeros((n_points_row, n_points_col))
  grid_loss = torch.zeros((n_points_row, n_points_col))
  origin_vector = unprojection(min_row, min_col)
  model_locations = list()
  # get the locations of the models on the grid
  for i1, model in enumerate(models_flattened):
    row, col = project_grid(origin_vector - average_model, model, scale_row, scale_col)
    row, col = (round(float(row), 2), round(float(col), 2))
    model_locations.append((row, col))

  # evaluate the grid
  for i1,t1 in enumerate(t1s):
    for i2,t2 in enumerate(t2s):
      new_flat_v = origin_vector + basis[0]*t1*scale_row + basis[1]*t2*scale_col
      reconstructed_flat = reconstruct(new_flat_v, models[0], keys)
      grid_acc[i1,i2], grid_loss[i1,i2] = eval_fn(reconstructed_flat)
  
  return {"grid_acc": grid_acc, "grid_loss": grid_loss, "model_locations": model_locations, "tick_rows": t1s, "tick_cols": t2s, "basis": basis, "scale_row": scale_row, "scale_col": scale_col}
