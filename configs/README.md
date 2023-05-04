# Configs
In this framework, configs define our experimental test suites. The name of each config describes the dataset the experiment will be run on, and the model it will use. It is important to follow the naming convention utilized in this directory, as our config parsing functions found in utils.py depend on this. 

## Fields
Each config looks like this: 
```python
config = {
    'dataset': {
        'name': "<DATASET_NAME>",               # name of the dataset. Should match a corresponding variable name found in datasets/config.py
        'train_fraction': None,                 # Optional. If you'd like to only use part of the training set, include this and the fraction. Otherwise delete it.
    },
    'model': {
        'name': '<MODEL_NAME>',                 # name of the model
        'dir': '<MODEL_CKPT_DIR>',              # checkpoint directory where models are stored
        'bases': []                             # list of optional model paths. Empty by default
    },
    'merging_fn': 'match_tensors_zipit',        # matching function desired. Please see "matching_functions.py" for a complete list of supported functions.
    'eval_type': 'clip',                        # Evaluation type, whether to use clip or standard cross entropy loss
    'merging_metrics': ['covariance', 'mean'],  # Alignment Metric types desired upon which to compute merging. Please see metric_calculators.py for more details
}
```

## New Configs
Adding new configs can be done simply by following the above format. 