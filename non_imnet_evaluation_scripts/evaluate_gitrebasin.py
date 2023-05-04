import os
import torch
import numpy as np
from utils import *
import pickle


def get_model_fname(model_idx, model_name=None, model_width=None):
    if model_name is None and model_width is None:
        pickle_name = f'{model_idx}.pkl'
    elif model_name is None:
        pickle_name = f'resnet20x{model_width}_{model_idx}.pkl'
    else:
        pickle_name = f'{model_name}_gitrebasin_{model_idx}.pkl'
    return pickle_name


def maybe_change_keys(sd): 
    key_replacer = lambda x: x.replace('params.', '').replace('batch_stats.', '').replace('scale', 'weight')
    new_sd = {}
    for k, v in sd.items():
        new_sd[key_replacer(k)] = v
    return new_sd
    

if __name__ == "__main__":
    config_name = 'cifar50_resnet20'
    checkpoint_dir = './checkpoints/cifar50/gitrebasins/logits'
    model_dir = './cifar50/pairs'
    
    SEED = 0
    set_seed(SEED)   
    model_idxs = np.arange(0, 4)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    raw_config['model']['dir'] = model_dir
    config = prepare_experiment_config(raw_config)
    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']

    run_pairs = find_runable_pairs(model_dir, config['model']['name'], skip_pair_idxs=[])

    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'gitrebasin.csv'
    )

    with torch.no_grad():
        # model = resnet20(w=model_width, num_classes=100)
        model = config['models']['new']
        for model_idx in model_idxs:
            model.load_state_dict(
                maybe_change_keys(
                    pickle.load(
                        open(
                            os.path.join(
                                checkpoint_dir, 
                                get_model_fname(
                                    model_idx, 
                                    model_name=config['model']['name']
                                    )
                                ), 
                            'rb'
                            )
                        )
                    )
                )
            model = model.to(device)
            model.eval()

            raw_config = inject_pair(raw_config, run_pairs[model_idx], ignore_bases=True)
            config = prepare_experiment_config(raw_config)
            
            results = evaluate_model(config['eval_type'], model, config)
            # pdb.set_trace()
            write_to_csv(results, csv_file)
            print(f'Model Results: {results}')