config = {
    'dataset': {
        'name': 'cifar5',
        'shuffle_train': True
    },
    'model': {
        'name': 'vgg11_w1',
        'dir': './checkpoints/cifar5_trainlogithead/vgg11_w1/pairsplits_ourinit_epochs100',
        'bases': []
    },
    'eval_type': 'logits',
    'merging_fn': 'match_tensors_zipit',
    'merging_metrics': ['covariance', 'mean'],
}