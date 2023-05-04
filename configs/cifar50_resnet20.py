config = {
    'dataset': {
        'name': 'cifar50',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet20x8',
        'dir': './checkpoints/cifar50_traincliphead/',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'eval_type': 'clip',
    'merging_metrics': ['covariance', 'mean'],
}