import torch
import torchvision
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imagenet_class import ImageNet
import os

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


def generate_random_class_splits(total_classes, split_proportions):
    splits = []
    start_idx = 0
    selection_indices = np.arange(total_classes)
    for i, split_prop in enumerate(split_proportions):
        if i == (len(split_proportions) - 1):
            splits += [selection_indices]
            break

        split_amount = int(total_classes * split_prop)
        split_idxs, selection_indices = train_test_split(selection_indices, train_size=split_amount)
        splits += [split_idxs]
    return splits

def split_even(total_classes, split_proportions):
    splits = [[]]

    for i in range(total_classes):
        if len(splits[-1]) >= int(total_classes * split_proportions[0]):
            del split_proportions[0]
            splits.append([])
        splits[-1].append(i)

    return splits


def create_subsets(train_dset, test_dset, model_class_splits):
    class_names = test_dset.classes

    model_loaders = []
    for i, model_classes in enumerate(model_class_splits):
        # Class indices
        if isinstance(model_classes[0], int) or isinstance(model_classes[0], np.int64):
            split_idxs = model_classes
        # Class names
        elif isinstance(model_classes[0], str):
            split_idxs  = [class_names.index(i) for i in model_classes]
        else:
            # pdb.set_trace()
            raise ValueError(f'unknown classes: {model_classes}')

        train_subset_idxs = [i for i, label in enumerate(train_dset.targets) if label in split_idxs]
        test_subset_idxs =  [i for i, label in enumerate(test_dset.targets) if label in split_idxs]

        train_subset = torch.utils.data.Subset(train_dset, train_subset_idxs)
        test_subset = torch.utils.data.Subset(test_dset, test_subset_idxs)

        model_loaders.append((train_subset, test_subset))
    return model_loaders


def write(dataset, path, name):
    print(f'writing {name}...')
    writer = DatasetWriter(os.path.join(path, name), {
        'image': RGBImageField(write_mode='smart',
                               max_resolution=500,
                               compress_probability=0.5,
                               jpeg_quality=90),
        'label': IntField(),
    }, num_workers=8)

    writer.from_indexed_dataset(dataset, chunksize=100)



if __name__ == "__main__":
    splits = (split_even(1000, [.2, .2, .2, .2, .2]))


    data_dir = '/srv/share4/datasets/ImageNet/imagenet'
    train_dset = ImageNet(root=data_dir, split='train')
    test_dset = ImageNet(root=data_dir, split='val')

    subsets = create_subsets(train_dset, test_dset, splits)

    out_dir = '/srv/flash2/dbolya3/ffcv/imagenet_splits/'

    for i, (train, val) in enumerate(subsets):
        write(train, out_dir, f'train_{i}_{len(train)}.ffcv')
        write(val, out_dir, f'val_{i}_{len(val)}.ffcv')
