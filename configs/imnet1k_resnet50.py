import numpy as np

config = {
    'dataset': {
        'name': 'imagenet1k',
        'train_fraction': 0.01,
        'inception_norm': False,
        # 'crop_ratio': 1.0,
        # 'res': 256,
    },
    'model': {
        'name': 'resnet50',
        'bases': [
            './checkpoints/resnet50_imagenet1k_1.pt', 
            './checkpoints/resnet50_imagenet1k_2.pt',
        ]
    },
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
}
