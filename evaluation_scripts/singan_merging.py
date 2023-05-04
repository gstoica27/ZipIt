import os
import pdb
import clip
import torch
import random
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model_merger import ModelMerge

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)


if __name__ == "__main__":
    with torch.no_grad():
        device = 0 # 'cuda' if torch.cuda.is_available() else 'cpu'
        config_name = 'singan'
        raw_config = get_config_from_name(config_name, device=device)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']

        # Construct Graphs
        Grapher = config['graph']
        graphs = []
        for base_model in tqdm(config['models']['bases'], desc="Creating Base Graphs: "):
            graphs.append(Grapher(base_model).graphify())

        base_sd = deepcopy(base_model.state_dict())

        # Construct Merger and Merge Models
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(config['models']['new'], train_loader, transform_fn=config['merging_fn'], metric_classes=config['metric_fns'])
        # Set New Model
        # reset_bn_stats(Merge, train_loader, reset=False)

        merge_sd = Merge.merged_model.state_dict()

        # for k, v in Merge.merged_model.amps.items():
        #     v.data /= len(config['models']['bases'])

        # for k, v in base_sd.items():
        #     if not torch.allclose(v, merge_sd[k], atol=1e-4):
        #         print(k)

        try:
            while True:
                Merge = Merge.eval()
                Merge.merged_model._save_image(Merge(torch.rand(1,1,1,1)))
        except KeyboardInterrupt:
            exit()
