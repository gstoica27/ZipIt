import os
import pdb
import torch
from copy import deepcopy

import numpy as np
from itertools import product
from utils import *
from model_merger import ModelMerge
from itertools import combinations

    
if __name__ == "__main__":
    EVAL_TYPE = 'hyperparameter_search'
    if EVAL_TYPE == 'hyperparameter_search':
        eval_split = 'heldout_val'
    else:
        eval_split = 'heldout_test'
        
    BIGSEED = 420
    set_seed(BIGSEED)
    
    config_name = 'multidataset_resnet50'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])

    dataset_pairs = combinations(np.arange(len(config['data'])), 2)
    print('Merging Function: {}'.format(config['merging_fn']))
    dataloaders = np.array([i for i in config['data']])
    
    csv_file = os.path.join(
        './csvs',
        ":".join(dataset_names),
        raw_config['model']['name'],
        raw_config['eval_type'],
        f'{EVAL_TYPE}_pairs.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
                
    search_config = {
        'a': [.7],
        'b': [0.15],
        'start_at': [None],
        'stop_at': [72],
    }        
    
    with torch.no_grad():
        models = np.array([i for i in config['models']['bases']])
        # dataloaders = [config['data'][i]['train']['full'] for i in range(len(raw_config['dataset']))]
        param_names, values = zip(*search_config.items())
        for pair in dataset_pairs:
            pair = list(pair)
            pair_loaders = dataloaders[pair]
            train_loaders = SpoofLoader(*[i['train']['full'] for i in pair_loaders]) 
            
            pair_models = models[pair]
            Grapher = config['graph']
            
            Merge = ModelMerge(device=device)
            for bundle in product(*values):
                instance_params = dict(zip(param_names, bundle))
                graphs =[Grapher(deepcopy(base_model)).graphify() for base_model in pair_models]    
                Merge.init(graphs, device=device)
                Merge.transform(
                    deepcopy(config['models']['new']), 
                    train_loaders,
                    transform_fn=config['merging_fn'], 
                    metric_classes=config['metric_fns'],
                    **instance_params
                )
                
                reset_bn_stats(Merge, train_loaders)
                pair_names = dataset_names[pair]
                results = deepcopy(instance_params)
                results.update({k: None for k in dataset_names})
                flops = Merge.compute_forward_flops()
                results['flops'] = flops
                print('FLOPS: {}'.format(flops))
                
                for i, loader_dict in enumerate(pair_loaders):
                    loader = loader_dict['test'][eval_split]
                    acc = evaluate_logits_i(Merge, loader, i, len(loader_dict['train']['full'].dataset.targets))
                    results[pair_names[i]] = acc
                write_to_csv(results, csv_file)
                print(results)
            