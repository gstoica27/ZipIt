import torch
import random
from copy import deepcopy

from tqdm.auto import tqdm
import numpy as np

from utils import *
from model_merger import ModelMerge

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def evaluate_logits_i(model, loader, splits, num_classes, return_confusion=True):
    model.eval()
    correct = 0
    total = 0
    
    splits = [list(split) for split in splits]

    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    all_splits = torch.tensor(sum(splits, [])).to(device)

    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i
    
    task_map = [task_map[_cls] if _cls in task_map else -1 for _cls in range(num_classes)]
    task_map = torch.tensor(task_map).to(device)

    splits = torch.tensor(splits).to(device)

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating multihead head model'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector, :, :, :], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]
            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = torch.stack(outputs, dim=1)
                outputs = outputs[range(batch_size), task_idx, :]

            task_splits = splits[task_idx, :]
            outputs = outputs.gather(dim=-1, index=task_splits).argmax(dim=-1)
            outputs = task_splits.gather(dim=-1, index=outputs[:, None])[:, 0]

            for gt, p in zip(labels, outputs):
                totals[gt] += 1
                
                if gt == p:
                    correct += 1
                    corrects[gt] += 1
                
                total += 1
    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / max(a[1], 1e-4), zip(corrects, totals)))
    else:
        return correct / total

model_splits = {
    2: [[0,4],[1,2],[1,4],[2,4]],
    3: [[0,1,4],[0,2,4],[1,2,3],[1,3,4]],
    4: [[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]],
    5: [[0,1,2,3,4]]
}

methods = ['match_tensors_identity', 'match_tensors_permute', 'match_tensors_zipit']

if __name__ == "__main__":
    with torch.no_grad():
        # stop_nodes = [159]
        stop_nodes = [None, 33, 72, 129, 159]
        device = 0
        config_name = 'imnet200_resnet50'

        for num_models, splits in model_splits.items():
            for run, split in enumerate(splits):
                if num_models < 3:# or (num_models == 3 and run < 3):
                    continue
                raw_config = get_config_from_name(config_name, device=device)

                raw_config['dataset']['class_splits'] = [raw_config['dataset']['class_splits'][i] for i in split]
                raw_config['model']['bases'] = [raw_config['model']['bases'][i] for i in split]
                config = prepare_experiment_config(raw_config)

                test_loader = config['data']['test']['full']
                splits = config['dataset']['class_splits']


                base_accs = []
                print('Evaluating base models...')
                for i, base_model in enumerate(config['models']['bases']):
                    reset_bn_stats(base_model, config['data']['train']['full'])
                
                # Construct Merger and Merge Models
                Merge = ModelMerge(device=device)

                for stop_node in stop_nodes:
                    for method in methods:
                        kwdargs = {}
                        if method == 'match_tensors_zipit':
                            if stop_node is None:
                                kwdargs['a'] = 0.5
                                kwdargs['b'] = 0.25
                            else:
                                kwdargs['a'] = 1.0
                                kwdargs['b'] = 0

                        else:
                            if stop_node is not None:
                                continue
                        # Construct Graphs
                        Grapher = config['graph']
                        graphs = []
                        for base_model in tqdm(config['models']['bases'], desc="Creating Base Graphs: "):
                            graphs.append(Grapher(deepcopy(base_model)).graphify())

                        Merge.init(graphs, device)
                        Merge.transform(
                            deepcopy(config['models']['new']), 
                            config['data']['train']['full'], 
                            transform_fn=get_merging_fn(method), 
                            metric_classes=config['metric_fns'],
                            stop_at=stop_node, # b=0.5, a=0.01
                        )
                        # Set New Model
                        reset_bn_stats(Merge, config['data']['train']['full'])
                        
                        acc = evaluate_logits_i(Merge, test_loader, splits=splits, num_classes=1000, return_confusion=False)

                        with open('imnet200_results.csv', 'a') as f:
                            # num_models,method,stop_at,run,acc
                            f.write(f"{num_models},{method},{run},{stop_node},{acc}\n")
