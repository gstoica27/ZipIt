import numpy as np

idx = [0, 1, 2, 3, 4]
# idx = [0, 1, 2, 3]
# idx = [1,2,4]

config = {
    'dataset': {
        'name': 'imagenet1k',
        'class_splits': [np.split(np.arange(1000), 5)[i] for i in idx],
        'train_fraction': 0.01,
    },
    'model': {
        'name': 'resnet50',
        'bases': [[
            './checkpoints/imnet200split/resnet50_0.pt',
            './checkpoints/imnet200split/resnet50_1.pt',
            './checkpoints/imnet200split/resnet50_2.pt',
            './checkpoints/imnet200split/resnet50_3.pt',
            './checkpoints/imnet200split/resnet50_4.pt',
        ][i] for i in idx]
    },
    'merging_fn': 'match_tensors_zipit',
    # 'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
    'finetune_epochs': 0,
    'train_epochs': 0
}
