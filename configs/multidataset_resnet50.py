config = {
    'dataset': [
        {
            'name': 'cub',
            'train_fraction': 0.5,
            'val_fraction':  .1,
            'shuffle_train': True,
            'crop_ratio': 1.0,
            # 'res': 256,
        },
        {
            'name': 'oxford_pets',
            'train_fraction': 0.1,
            'val_fraction':  .1,
            'shuffle_train': True,
        },
        {
            'name': 'stanford_dogs',
            'train_fraction': 0.1,
            'val_fraction':  .1,
            'shuffle_train': True,
            'crop_ratio': 1.0,
            # 'res': 256,
        },
        {
            'name': 'nabird',
            'train_fraction': 0.1,
            'val_fraction':  .1,
            'shuffle_train': True
        },
    ],
    'model': {
        'name': 'resnet50',
        'bases': [
            # './checkpoints/multiset/stanford_dogs/resnet50_from_imagenet1k_v2_v0.pth.tar'
            './checkpoints/multiset/cub/resnet50_from_daniel_ckpt2_v0.pth.tar',
            './checkpoints/multiset/oxford_pets/resnet50_from_imagenet1k_v1_v0.pth.tar',
            './checkpoints/multiset/stanford_dogs/resnet50_from_daniel_ckpt1_v0.pth.tar', 
            '/srv/share/rmittapalli3/model_finder/trained/resnet50_models/resnet50_nabird.pth',
        ]
    },
    'eval_type': 'logits',
    # 'merging_fn': 'match_tensors_identity',
    'merging_fn': 'match_tensors_zipit',
    # 'merging_fn': 'match_tensors_permute',
    # 'merging_fn': 'match_tensors_zipit_greedy',
    'merging_metrics': ['covariance', 'mean'],
    'finetune_epochs': 0
}
