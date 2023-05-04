from collections import defaultdict
from copy import deepcopy
import pdb
import torch
from torch import nn
from fvcore.nn.flop_count import flop_count
from graphs.base_graph import NodeType
from metric_calculators import CovarianceMetric, MeanMetric
from matching_functions import match_tensors_zipit
from time import time
from tqdm.auto import tqdm
from utils import get_merging_fn


class MergeHandler:
    """ 
    Handles all (un)merge transformations on top of a graph architecture. merge/unmerge is a dict whose 
    keys are graph nodes and values are merges/unmerges to be applied at the graph node.
    """
    def __init__(self, graph, merge, unmerge):
        self.graph = graph
        # (Un)Merge instructions for different kinds of module layers.
        self.module_handlers = {
            'BatchNorm2d': self.handle_batchnorm2d,
            'Conv2d': self.handle_conv2d,
            'Linear': self.handle_linear,
            'LayerNorm': self.handle_layernorm,
            'GELU': self.handle_fn,
            'AdaptiveAvgPool2d': self.handle_fn,
            'LeakyReLU': self.handle_fn,
            'ReLU': self.handle_fn,
            'Tanh': self.handle_fn,
            'MaxPool2d': self.handle_fn,
            'AvgPool2d': self.handle_fn,
             'SpaceInterceptor': self.handle_linear,
             'Identity': self.handle_fn
        }

        self.merge = merge
        self.unmerge = unmerge
    
    def handle_batchnorm2d(self, forward, node, module):
        """ Apply (un)merge operation to batchnorm parameters. """
        if forward:
            # Forward will always be called on a batchnorm, but backward might not be called
            # So merge the batch norm here.
            for parameter_name in ['weight', 'bias', 'running_mean', 'running_var']:
                parameter = getattr(module, parameter_name)
                merge = self.merge if parameter_name != 'running_var' else self.merge # ** 2
                parameter.data = merge @ parameter
            
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, 'BN expects one predecessor'
            self.prop_back(self.graph.preds(node)[0])
    
    def handle_layernorm(self, forward, node, module):
        """ Apply (un)merge operation to layernorm parameters. """
        if forward:
            # Forward will always be called on a norm, so merge here
            parameter_names = ['weight', 'bias']
            for parameter_name in parameter_names:
                parameter = getattr(module, parameter_name)
                parameter.data = self.merge @ parameter
            
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, 'LN expects one predecessor'
            self.prop_back(self.graph.preds(node)[0])

    def handle_fn(self, forward, node, module):
        """ Apply (un)merge operation to parameterless layers. """
        if forward:
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, 'Function node expects one predecessor'
            self.prop_back(self.graph.preds(node)[0])

    def handle_conv2d(self, forward, node, module):
        """ Apply (un)merge operation to linear layer parameters. """
        if forward: # unmerge
            try:
                module.weight.data = torch.einsum('OIHW,IU->OUHW', module.weight, self.unmerge)
            except:
                pdb.set_trace()
        else: # merge
            try:
                module.weight.data = torch.einsum('UO,OIHW->UIHW', self.merge, module.weight)
            except:
                pdb.set_trace()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = self.merge @ module.bias
    
    def handle_linear(self, forward, node, module):
        """ Apply (un)merge operation to linear layer parameters. """
        if forward: # unmerge
            module.weight.data = module.weight @ self.unmerge
        else:
            info = self.graph.get_node_info(node)

            lower = 0
            upper = module.weight.shape[0]

            if info['chunk'] is not None:
                idx, num_chunks = info['chunk']
                chunk_size = upper // num_chunks

                lower = idx * chunk_size
                upper = (idx+1) * chunk_size

            module.weight.data[lower:upper] = self.merge @ module.weight[lower:upper]
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[lower:upper] = self.merge @ module.bias[lower:upper]

    def prop_back(self, node):
        """ Propogate (un)merge metrics backwards through a node graph. """
        if node in self.graph.merged:
            return
        
        info = self.graph.get_node_info(node)
        self.graph.merged.add(node)
        
        for succ in self.graph.succs(node):
            self.prop_forward(succ)
        
        if info['type'] in (NodeType.OUTPUT, NodeType.INPUT):
            raise RuntimeError(f'Unexpectedly reached node type {info["type"]} when merging.')
        elif info['type'] == NodeType.CONCAT:
            # Also this only works if you concat the same size things together
            merge = self.merge.chunk(len(self.graph.preds(node)), dim=1)
            for pred, m in zip(self.graph.preds(node), merge):
                MergeHandler(self.graph, m, self.unmerge).prop_back(pred)
        elif info['type'] == NodeType.MODULE:
            module = self.graph.get_module(info['layer'])
            # try:
            self.module_handlers[module.__class__.__name__](False, node, module)
            # except:
            #     pdb.set_trace()
        elif info['type'] == NodeType.EMBEDDING:
            param = self.graph.get_parameter(info['param'])
            self.handle_embedding(False, node, param)
        else:
            # Default case (also for SUM)
            for pred in self.graph.preds(node):
                self.prop_back(pred)
    
    def prop_forward(self, node):
        """ Propogate (un)merge transformations up a network graph. """
        if node in self.graph.unmerged:
            return
        
        info = self.graph.get_node_info(node)
        self.graph.unmerged.add(node)
        
        if info['type'] in (NodeType.OUTPUT, NodeType.INPUT):
            raise RuntimeError(f'Unexpectedly reached node type {info["type"]} when unmerging.')
        elif info['type'] == NodeType.MODULE:
            module = self.graph.get_module(info['layer'])
            self.module_handlers[module.__class__.__name__](True, node, module)
        elif info['type'] == NodeType.SUM:
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
            for pred in self.graph.preds(node):
                self.prop_back(pred)
        elif info['type'] == NodeType.CONCAT:
            # let's make the assumption that this node is reached in the correct order
            num_to_concat = len(self.graph.preds(node))

            if node not in self.graph.working_info:
                self.graph.working_info[node] = []
            self.graph.working_info[node].append(self.unmerge)
            
            if len(self.graph.working_info[node]) < num_to_concat:
                # haven't collected all the info yet, don't finish the unmerge
                self.graph.unmerged.remove(node)
            else:
                # finally, we're finished
                unmerge = torch.block_diag(*self.graph.working_info[node])
                del self.graph.working_info[node]

                # be free my little unmerge
                new_handler = MergeHandler(self.graph, self.merge, unmerge)
                for succ in self.graph.succs(node):
                    new_handler.prop_forward(succ)
                



class MergedModelStop(Exception):
    """ Custom error handle to catch point when forward pass enters each base model head at partial zipping. """
    def __init__(self, x):
        self.x = x



class ModelMerge(nn.Module):
    """
    Handles all merge operations for zipping arbitrary numbers of models. 
    Expects a list of architecture graphs (one per model) (See graphs/base_graphs.py)).
    """
    def __init__(self, *graphs, device=0):
        super().__init__()
        
        self.stop_at = None
        self.start_at = None
        self.stop_at_ptr = [None]
        self.start_at_ptr = {}

        self.hooks = []

        self.init(graphs, device)

    def init(self, graphs, device):
        """
        Initialize merge attributes with new set of graphs.
        """
        # move all graph models to eval
        for g in graphs:
            g.model.to(device).eval()

        self.graphs = graphs
        self.device = device

        self.merged_model = None
        # Initialize heads for partial zipping
        self.head_models = nn.ModuleList([g.model for g in self.graphs])
        # Add hooks on intermediate layers for computing intra-model alignment metrics
        for graph in self.graphs:
            graph.add_hooks(device=device)


    def compute_metrics(self, dataloader, metric_classes):
        """
        Compute pairwise alignment metrics between all graph models (self inclusive).
        - dataloader: pytorch dataloader. Dataset (or list of datasets) over which to compute metrics
        - metric_classes: dictionary whose keys are metric types, and values are metric functions.
            This function will compute metrics for all kinds in metric_classes, using the dataloader.
        
        This function performs a forward pass over the dataset, aggregating all intermediate representations
        among all hooks in a graph model. These are then combined to calculate metrics.    
        
        Returns: dictionary of graph nodes to metrics computed at those nodes in the model graph.
        """
        self.metrics = None
        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        
        numel = 0
        for dataloader in dataloader_list:
            for x, _ in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):
                x = x.to(self.device)
                
                numel += x.shape[0]
                intermediates = [g.compute_intermediates(x) for g in self.graphs]
                nodes = list(intermediates[0].keys())
                if self.metrics is None:
                    self.metrics = {n: {k: v() for k, v in metric_classes.items()} for n in nodes}
                
                for node, node_metrics in self.metrics.items():
                    for metric in node_metrics.values():
                        intermeds_float = [i[node].float() for i in intermediates]
                        metric.update(x.shape[0], *intermeds_float)
        
        for node, node_metrics in self.metrics.items():
            for metric_name, metric in node_metrics.items():
                self.metrics[node][metric_name] = metric.finalize(numel)

        return self.metrics
    
    def compute_transformations(self, transform_fn, reduce_ratio=.5, **kwargs):
        """
        Transforms graph models according to a transform function (transform_fn) using the alignment 
        metrics provided by self.metrics. Will transform the feature spaces at each PREFIX and POSTFIX 
        node between all models. The objective of this operation is to map all dispirate feature spaces 
        in each model graph to a common one such that all distinct spaces collectively map to a single 
        space of dimension (1 - reduce_ratio) * sum(graph1_feat_dim + graph2_feat_dim + ... + graphN_feat_dim)
        - transform_fn: transformation function (e.g., permutation - match_tensors_permute)
        - reduce_ratio: desired reduction proportion from total of all graph model feature dimensions
        - kwargs: hyperparameters associated with transform_fn. E.g., alpha and beta for ZipIt!
        Returns: A dictionay for transform operations to be performed at every point defined by PREFIX and POSTFIX, 
        on all graph models.
        """
        start_time = time()
        self.merges = {}
        self.unmerges = {}
        
        nodes = list(self.metrics.keys())
        nodes.sort()

        for node in tqdm(nodes, desc="Computing transformations: "):
            if self.start_at is None or node >= self.start_at:
                metric = self.metrics[node]
                # Maybe merge differently 
                info = self.graphs[0].get_node_info(node)
                if info['special_merge'] is not None:
                    merge, unmerge = get_merging_fn(info['special_merge'])(metric, reduce_ratio, **kwargs)
                else:
                    merge, unmerge = transform_fn(metric, reduce_ratio, **kwargs)
                
                # TODO: check if better way to do hack
                merge = merge * len(self.graphs) # Hack to deal with things not merged
                
                self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
                
                if self.stop_at is not None and node == self.stop_at:
                    break
        
        self.compute_transform_time = time() - start_time
        return self.merges, self.unmerges
    
    def apply_transformations(self):
        """
        Applys transformations found by compute_transformations from start_at up to stop_at graph node location 
        on all graph models. 
        """
        for node in self.merges:
            merges = self.merges[node]
            unmerges = self.unmerges[node]
            for merge, unmerge, graph in zip(merges, unmerges, self.graphs):
                merger = MergeHandler(graph, merge, unmerge)
                merger.prop_back(node)
        
    def get_merged_state_dict(self, interp_w=None):
        """
        Post transformations, obtain state dictionary for merged model by linearly interpolating between 
        transformed models in each graph. By default all parameters are averaged, but if given an interp_w 
        weight, will be weightedly averaged instead.
        - interp_w (Optional): If None, all parameters of each model is averaged for merge. Otherwise, 
        interp_w is a list of len(num_models_to_merge), with weights bearing the importance of incorporating 
        features from each model into the merged result.
        Returns: state dict of merged model.
        """
        state_dict = {}
        merged_state_dict = self.merged_model.state_dict()
        keys = list(self.graphs[0].model.state_dict().keys())
        try:
            for key in keys:
                if key in merged_state_dict:
                    param = self.graphs[0].model.state_dict()[key]
                    if interp_w is not None and param.shape == merged_state_dict[key].shape:
                        new_value = sum(graph.model.state_dict()[key] * w for graph, w in zip(self.graphs, interp_w))
                    else:
                        new_value = sum(graph.model.state_dict()[key] for graph in self.graphs) / len(self.graphs)
                    state_dict[key] = new_value
        except RuntimeError as e:
            # Only catch runtime errors about tensor sizes, we need to be able to add models with diff heads together
            if 'size' not in str(e):
                raise e
        return state_dict
    
    
    def add_prop_hook(self, model, node, pre=False, stop=False, loc=None, loc_idx=0, tmp_dict=None, tmp_dict_size=1):
        """
        Helper used for partial zipping. Add forward propogation hooks to grab intermediate outputs wherever partial zipping starts/stops. 
        These iintermediate outputs of each base model/merged model respectively will then be passed to the merged model/base models 
        respectivelty.
        """
        info = self.graphs[0].get_node_info(node)
        module = dict(model.named_modules())[info['layer']]

        def process(x):
            if len(tmp_dict) >= tmp_dict_size:
                tmp_dict.clear()
            tmp_dict[loc_idx] = x

            if len(tmp_dict) >= tmp_dict_size:
                raise MergedModelStop(tmp_dict)

            return None

        if pre:
            def prehook(m, x):
                if stop:
                    return process(x[0])
                else:
                    return loc[loc_idx]
            self.hooks.append(module.register_forward_pre_hook(prehook))
        else:
            def posthook(m, x, y):
                if stop:
                    return process(y)
                else:
                    return loc[loc_idx]
            self.hooks.append(module.register_forward_hook(posthook))

    def has_weight_matrix(self, node):
        """ Whether a graph node has an associated weight matrix (i.e., whether it has parameters needing to be transformed). """
        
        
        info = self.graphs[0].get_node_info(node)

        if info['type'] == NodeType.MODULE:
            _cls = self.graphs[0].get_module(info['layer']).__class__.__name__
            if _cls in ('Linear', 'Conv2d', 'Conv1d', 'Conv3d', 'SpaceInterceptor'):
                return True
        
        return False

    def add_unmerge_hooks(self, model_stop, models_start, loc):
        """ Finds every weight module that was unmerged but not merged. """
        graph = self.graphs[0]
        tmp_dict = {}

        nodes = [node for node in graph.G if
                 node not in graph.merged
                 and node in graph.unmerged
                 and self.has_weight_matrix(node)]
        
        for idx, node in enumerate(nodes):
            self.add_prop_hook(model_stop, node, pre=True, stop=True, loc_idx=0, tmp_dict=tmp_dict, tmp_dict_size=1)
            for model in models_start:
                self.add_prop_hook(model, node, pre=True, stop=False, loc=loc, loc_idx=0)

    def add_merge_hooks(self, model_stop, models_start, loc):
        """ Finds the first weight module that was merged but not unmerged. """
        graph = self.graphs[0]
        tmp_dict = {}

        nodes = [node for node in graph.G if
                 node not in graph.unmerged
                 and node in graph.merged
                 and self.has_weight_matrix(node)]

        for idx, node in enumerate(nodes):
            self.add_prop_hook(model_stop, node, pre=False, stop=False, loc=loc, loc_idx=idx)
            for model in models_start:
                self.add_prop_hook(model, node, pre=False, stop=True, loc_idx=idx, tmp_dict=tmp_dict, tmp_dict_size=len(nodes))

    def clear_hooks(self):
        """ Clears all hooks from graphs. """
        for g in self.graphs:
            g.clear_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []      
            
    def transform(self, model,
                  dataloader,
                  metric_classes=(CovarianceMetric, MeanMetric),
                  transform_fn=match_tensors_zipit,
                  prune_threshold=0.,
                  stop_at=None,
                  start_at=None,
                  **transform_kwargs
                  ):
        """ Note: this consumes the models given to the graphs. Do not modify the models you give this. """
        
        self.stop_at = stop_at
        self.start_at = start_at
        self.merged_model = model.to(self.device)
        
        if not isinstance(metric_classes, dict):
            metric_classes = { x.name: x for x in metric_classes }
        
        self.metric_classes = metric_classes
        self.transform_fn = transform_fn
        self.prune_threshold = prune_threshold
        
        self.compute_metrics(dataloader, metric_classes=metric_classes)
        self.compute_transformations(transform_fn,
                                    reduce_ratio=1 - 1. / len(self.graphs),
                                    prune_threshold=prune_threshold,
                                    **transform_kwargs
                                    )
        self.apply_transformations()
        
        self.merged_model.load_state_dict(self.get_merged_state_dict(), strict=False)
        
        self.add_hooks()
    
    def add_hooks(self):
        """ Add hooks at zip start or stop at locations for merged model and base models. """
        # Remove the hooks from the models to add or own
        self.clear_hooks()
        
        if self.stop_at is not None:
            self.add_unmerge_hooks(self.merged_model, self.head_models, self.stop_at_ptr)
        if self.start_at is not None:
            self.start_at_models = [deepcopy(g.model) for g in self.graphs]
            self.add_merge_hooks(self.merged_model, self.start_at_models, self.start_at_ptr)


    def add_head(self, graph_fn, model, dataloader):
        """ Add a new head to this merge assuming you've already called transform once. """
        assert self.stop_at is None and self.start_at is None, "multihead no work yet"

        self.clear_hooks()
        old_graph = graph_fn(self.merged_model).graphify()
        new_graph = graph_fn(model).graphify()

        old_graphs = [deepcopy(old_graph)]

        self.init(old_graphs + [new_graph], device=self.device)

        self.merged_model = old_graph.model.to(self.device)
        self.compute_metrics(dataloader, metric_classes=self.metric_classes)
        self.compute_transformations(self.transform_fn,
                                     1 - 1. / len(self.graphs),
                                     prune_threshold=self.prune_threshold)
        self.apply_transformations()

        self.merged_model.load_state_dict(self.get_merged_state_dict(), strict=False)

        self.clear_hooks()

    def forward(self, x, cat_dim=None, start_idx=None):
        """ Evaluate the combined model. """
        if self.start_at is not None:
            start_val = defaultdict(lambda: 0)
            total = 0

            for idx, model in enumerate(self.start_at_models):
                if start_idx is not None and idx != start_idx:
                    continue

                try:
                    model(x)
                except MergedModelStop as e:
                    for k, v in e.x.items():
                        start_val[k] = start_val[k] + v
                    total += 1
            
            self.start_at_ptr.clear()
            for k, v in start_val.items():
                self.start_at_ptr[k] = v / total / len(self.graphs)
            x = x[0, None].detach()
        
        try:
            return self.merged_model(x)
        except MergedModelStop as e:
            self.stop_at_ptr[0] = e.x[0]

            dummy_x = x[0, None].detach()
            out = []
            for idx, model in enumerate(self.head_models):
                out.append(model(dummy_x))

            self.stop_at_ptr[0] = None
            
            if cat_dim is not None:
                out = torch.cat(out, dim=cat_dim)
            
            return out
    
    def compute_flops(self, model, input_size=(3, 224, 224), batch_size=1):
        """ Compute the flops of a given model in eval mode. """
        model = model.eval().to(self.device)

        input1 = torch.rand(batch_size, *input_size, device=self.device)

        count_dict1, *_ = flop_count(model, input1)
        count1 = sum(count_dict1.values())
            
        return count1
    
    def compute_forward_flops(self, input_size=(3, 224, 224), cat_dim=None, start_idx=None):
        """ Evaluate the combined model. """
        # Note: does not support start_at yet

        if self.stop_at is None:
            return self.compute_flops(self.merged_model, input_size=input_size)
        else:
            dummy = torch.randn(len(self.head_models)-1, *input_size, device=self.device)
            try:
                self.merged_model(dummy)
            except MergedModelStop as e:
                self.stop_at_ptr[0] = e.x[0]
            
                # 1 base model
                self.clear_hooks()
                flops = self.compute_flops(self.graphs[0].model, input_size=input_size, batch_size=1)

                # n-1 head models once stop_at_ptr is reached
                self.add_hooks()
                flops += self.compute_flops(self.head_models[0], input_size=input_size, batch_size=1)

                self.stop_at_ptr[0] = None
                # base model + (n-1)*head models
                return flops
