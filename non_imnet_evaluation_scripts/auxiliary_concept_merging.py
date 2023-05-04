import os
import torch
import random

from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np

from utils import *
from model_merger import ModelMerge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def run_auxiliary_experiment(merging_fn, experiment_config, pairs, device, stop_at=None, csv_file=''):
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(experiment_config)
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        
        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(
            deepcopy(config['models']['new']), 
            train_loader, 
            transform_fn=get_merging_fn(merging_fn), 
            metric_classes=config['metric_fns'],
            stop_at=stop_at,
        )
        reset_bn_stats(Merge, train_loader)
        
        results = evaluate_model(experiment_config['eval_type'], Merge, config)
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = merging_fn
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = split
        write_to_csv(results, csv_file=csv_file)
        print(results)
        
    return results


if __name__ == "__main__":
    # config_name = 'cifar5_vgg'
    config_name = 'cifar50_resnet20'
    skip_pair_idxs = [0]
    merging_fns = [
        'match_tensors_permute',
        'match_tensors_identity',
    ]
    stop_at = 63
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    
    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'auxiliary_functions.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with torch.no_grad():
        for merging_fn in merging_fns:
            node_results = run_auxiliary_experiment(
                merging_fn=merging_fn, 
                experiment_config=raw_config, 
                pairs=run_pairs, 
                device=device, 
                csv_file=csv_file,
                stop_at=stop_at
            )