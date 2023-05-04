config = {
    'dataset': {
        'name': 'cifar10',
        'train_fraction': 0.01,
        'batch_size': 10,
    },
    'model': {
        'name': 'singan',
        'bases': [
            './checkpoints/singan/v4/balloon_models',
            # './checkpoints/singan/birds/',
            './checkpoints/singan/v4/bird_models',
        ]
    },
    'merging_fn': 'match_tensors_permute',
    # 'merging_fn': 'match_tensors_zipit',
    # 'merging_fn': 'match_tensors_pairs_lsa',
    # 'merging_fn': 'match_tensors_identity',
    'merging_metrics': ['covariance', 'mean'],
    'finetune_epochs': 0,
    'eval_type': 'logits'
}
