import os
import pdb
import torch
import random
from copy import deepcopy

import numpy as np
from itertools import product

from utils import *
from model_merger import ModelMerge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def hyperparam_search(
    config, stop_nodes, 
    search_config, 
    device, 
    csv_file=''
    ):
    keys, values = zip(*search_config.items())
    Merge = ModelMerge(device=device)
    for stop_node in stop_nodes:
        for bundle in product(*values):
            instance_params = dict(zip(keys, bundle))
        
            Grapher = config['graph']
            graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
            # Construct Merger and Merge Models
            # Merge = ModelMerge(*graphs, device=device)
            Merge.init(graphs, device=device)
            Merge.transform(
                deepcopy(config['models']['new']), 
                train_loader, 
                transform_fn=config['merging_fn'], 
                metric_classes=config['metric_fns'],
                stop_at=stop_node,
                **instance_params
            )
            reset_bn_stats(Merge, train_loader)
            # Create Results Dict
            results = evaluate_model(config['eval_type'], Merge, config)
            results.update(flatten_nested_dict(instance_params, parent_key='params', sep=' '))
            results['stop_node'] = stop_node
            results['Merge Fn'] = config['merging_fn'].__name__
            results['Time'] = Merge.compute_transform_time
            print(results)
            write_to_csv(results, csv_file=csv_file)
            

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = 'cifar5_resnet20'
    eval_pair = 0
    
    stop_nodes = [21]
    search_config = {
        'a':[.0001, .01, .1, .3, .5, 1.][::-1],
        'b': [.0, .0375, .075, .125, .25, 1.][::-1]
    }
    
    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=[])
    pair = run_pairs[eval_pair]
    # pdb.set_trace()
    print(f'PAIR: {pair}')
    raw_config = inject_pair(raw_config, pair)
    config = prepare_experiment_config(raw_config)
    
    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'zipit_hyperparameters.csv'
    )
    
    with torch.no_grad():
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        
        import pickle
        evaled_models = [pickle.loads(pickle.dumps(base_model)) for base_model in base_models]
        for i, base_model in enumerate(evaled_models):
            base_acc = evaluate_logits(
                base_model, config['data']['test']['splits'][i], use_flip_aug=False, 
            )
            print('Base Model {} Acc: {}'.format(i, base_acc))
        
        for b, e in zip(base_models, evaled_models):
            print([k for (k, v) in b.state_dict().items() if (v != e.state_dict()[k]).sum() > 0])
        
        hyperparam_search(
            config=config, 
            stop_nodes=stop_nodes,
            search_config=search_config,
            device=device,
            csv_file=csv_file
        )