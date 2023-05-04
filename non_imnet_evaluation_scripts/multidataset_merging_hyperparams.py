import os
import pdb
import clip
import torch
import random
from copy import deepcopy

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utils import *
from model_merger import ModelMerge


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
    datasets = ':'.join([i['name'] for i in raw_config['dataset']])
    
    csv_file = os.path.join(
        './csvs',
        datasets,
        raw_config['model']['name'],
        raw_config['eval_type'],
        f'{EVAL_TYPE}_all.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    search_config = {
        'a': np.arange(.0, 1.05, .05),
        'b': np.arange(0., 1.05, .05),
        'start_at': [None],
        'stop_at': [72],
    }        
    
    print('Merging Function: {}'.format(config['merging_fn']))
    dataloaders = [config['data'][i]['train']['full'] for i in range(len(raw_config['dataset']))]
    joint_dataloaders = SpoofLoader(*dataloaders)
    
    with torch.no_grad():
        Merge = ModelMerge(device=device)
        param_names, values = zip(*search_config.items())
        for bundle in product(*values):
            instance_params = dict(zip(param_names, bundle))
            Grapher = config['graph']
            graphs = []
            for i, base_model in enumerate(config['models']['bases']):
                graphs.append(Grapher(deepcopy(base_model)).graphify())
            Merge.init(graphs, device=device)
            Merge.transform(
                deepcopy(config['models']['new']), 
                joint_dataloaders,
                transform_fn=config['merging_fn'], 
                metric_classes=config['metric_fns'],
                **instance_params
            )
            reset_bn_stats(Merge, joint_dataloaders)
            
            results = {}
            results.update(instance_params)
            for i, loader_dict in enumerate(config['data']):
                loader = loader_dict['test'][eval_split]
                acc = evaluate_logits_i(Merge, loader, i, len(loader_dict['train']['full'].dataset.targets))
                results[raw_config['dataset'][i]['name'] + ' Acc'] = acc
            print(results)
            write_to_csv(results, csv_file)