config = {
    'dataset': {
        'name': 'cifar50',
        'shuffle_train': True
    },
    'model': {
        'name': 'vgg11',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'merging_metrics': ['covariance', 'mean'],
}


