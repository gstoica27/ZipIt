config = {
    'dataset': {
        'name': 'cifar5',
    },
    'model': {
        'name': 'resnet20x4',
        'dir': './checkpoints/cifar5_traincliphead/',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'eval_type': 'clip',
    'merging_metrics': ['covariance', 'mean'],
}