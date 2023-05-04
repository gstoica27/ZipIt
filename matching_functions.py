import torch
import scipy
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, KMeans
import pdb

#####################################################################################################################################
############################################################## HELPERS ##############################################################
#####################################################################################################################################

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def compute_correlation(covariance, eps=1e-7):
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance

def add_bias_to_mats(mats):
    """ Maybe add bias to input. """
    pad_value = 0
    pad_func = torch.nn.ConstantPad1d((0, 1, 0, 1), pad_value)
    biased_mats = []
    for mat in mats:
        padded_mat = pad_func(mat)
        padded_mat[-1, -1] = 1
        biased_mats.append(padded_mat)
    return biased_mats


#####################################################################################################################################
#################################################### MATCHING/ALIGNMENT FUNCTIONS ###################################################
#####################################################################################################################################

def match_tensors_zipit(
    metric, r=.5, a=0.3, b=.125, 
    print_merges=False, get_merge_value=False, add_bias=False, 
    **kwargs
):
    """
    ZipIt! matching algorithm. Given metric dict, computes matching as defined in paper. 
    Args:
    - metric: dictionary containing metrics. This must contain either a covariance or cossim matrix, and 
        must be [(num_models x model_feature_dim), (num_models x model_feature_dim)]. 
    - r: Amount to reduce total input feature dimension - this is num_models x model_feature_dim. This function will
        compute (un)merge matrix that goes from 
        (num_models x model_feature_dim) -> (1-r)*(num_models x model_feature_dim) = merged_feature_dim.
        E.g. if num_models=2, model_feature_dim=10 and r=.5, the matrix will map from 2x10=20 -> (1-.5)x2x10=10, or halve the 
        collective feature space of the models.
    - a: alpha hyperparameter as defined in Section 4.3 of our paper. 
    - b: beta hyperparameter as defined in Section 4.3 of our paper.
    - print_merges: whether to print computed (un)merge matrices.
    - get_merge_value default False, returns the sum of correlations over all the merges which the algorithm made. 
    - add_bias: whether to add a bias to the input. This should only be used if your module expects the input with bias offset.
    returns:
    - (un)merge matrices
    """
    if "covariance" in metric:
        sims = compute_correlation(metric["covariance"])
    elif "cossim" in metric:
        sims = metric["cossim"]
    O = sims.shape[0]
    remainder = int(O * (1-r) + 1e-4)
    permutation_matrix = torch.eye(O, O)#, device=sims.device)

    torch.diagonal(sims)[:] = -torch.inf

    num_models = int(1/(1 - r) + 0.5)
    Om = O // num_models

    original_model = torch.zeros(O, device=sims.device).long()
    for i in range(num_models):
        original_model[i*Om:(i+1)*Om] = i

    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long() + int((to_remove // num_models) * b + 1e-4)

    merge_value = []

    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        
        merge_value.append(permutation_matrix[row_idx, col_idx])

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        row_origin = original_model[row_idx]
        col_origin = original_model[col_idx]
        
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx)
        
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        
        if 'magnitudes' in metric:
            metric['magnitudes'][row_idx] = torch.minimum(metric['magnitudes'][row_idx], metric['magnitudes'][col_idx])
            metric['magnitudes'] = remove_col(metric['magnitudes'][None], col_idx)[0]
        
        if a <= 0:
            sims[row_origin*Om:(row_origin+1)*Om, row_idx] = -torch.inf
            sims[col_origin*Om:(col_origin+1)*Om, row_idx] = -torch.inf
        else: sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx)
        
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        if a <= 0:
            sims[row_idx, row_origin*Om:(row_origin+1)*Om] = -torch.inf
            sims[row_idx, col_origin*Om:(col_origin+1)*Om] = -torch.inf
        else: sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx).T

        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]
        
        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1

            if budget[origin] <= 0:
                # kill origin
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf
    
    if add_bias:
        unmerge_mats = permutation_matrix.chunk(num_models, dim=0)
        unmerge_mats = add_bias_to_mats(unmerge_mats)
        unmerge = torch.cat(unmerge_mats, dim=0)
    else:
        unmerge = permutation_matrix

    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
    if print_merges:
        O, half_O = unmerge.shape
        A_merge, B_merge = unmerge.chunk(2, dim=0)
        
        A_sums = A_merge.sum(0)
        B_sums = B_merge.sum(0)
        
        A_only = (B_sums == 0).sum()
        B_only = (A_sums == 0).sum()
        
        overlaps = half_O - (A_only + B_only)
        
        print(f'A into A: {A_only} | B into B: {B_only} | A into B: {overlaps}')
        print(f'Average Connections: {unmerge.sum(0).mean()}')
    
    merge = merge.to(sims.device)
    unmerge = unmerge.to(sims.device)
    if get_merge_value:
        merge_value = sum(merge_value) / len(merge_value)
        return merge.T, unmerge, merge_value
    return merge.T, unmerge


def match_tensors_optimal(metric, r=.5, add_bias=False, **kwargs):
    """ 
    Apply optimal algorithm to compute matching. 
    Hyperparameters and return are as defined in match_tensors_zipit.
    """
    correlation = metric["covariance"]
    corr_mtx_a = compute_correlation(correlation).cpu().numpy()
    min_num = corr_mtx_a[corr_mtx_a != -torch.inf].min() - 1
    G = nx.Graph()
    O = corr_mtx_a.shape[0] // 2
    for i in range(2*O):
        G.add_node(i)
    for i in range(2*O):
        for j in range(0, i):
            G.add_edge(i, j, weight=(corr_mtx_a[i, j] - min_num))
    matches = nx.max_weight_matching(G)
    matches_matrix = torch.zeros(2 * O, O, device=correlation.device)
    for i, (a, b) in enumerate(matches):
        matches_matrix[a,i] = 1
        matches_matrix[b,i] = 1
    merge = matches_matrix / (matches_matrix.sum(dim=0, keepdim=True) + 1e-5)
    unmerge = matches_matrix
    return merge.T, unmerge


def match_tensors_permute(metric, r=.5, get_merge_value=False, add_bias=False, **kwargs):
    """
    Matches arbitrary models by permuting all to the space of the first in your graph list. 
    Mimics Rebasin methods. 
    Hyperparameters and return are as defined in match_tensors_zipit.
    """
    correlation = compute_correlation(metric["covariance"])
    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    
    mats = [torch.eye(Om, device=device)]
    # mats = [torch.zeros(Om, Om, device=device)]
    for i in range(1, N):
        try:
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                correlation[:Om, Om*i:Om*(i+1)].cpu().numpy(), maximize=True)
        except:
            pdb.set_trace()
        mats.append(torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)].T)
    
    if add_bias:
        unmerge_mats = add_bias_to_mats(mats)
    else:
        unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if get_merge_value:
        merge_value = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()[row_ind, col_ind].mean()
        return merge.T, unmerge, merge_value
    return merge.T, unmerge


def match_tensors_identity(metric, r=.5, add_bias=False, **kwargs):
    """ 
    Match feature spaces from different models by weight averaging.  
    Hyperparameters and return are as defined in match_tensors_zipit.
    """
    correlation = metric["covariance"]
    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    
    mats = [torch.eye(Om, device=device) for _ in range(N)]
    
    if add_bias:
        unmerge_mats = add_bias_to_mats(mats)
    else:
        unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge


def match_tensors_kmeans(metric, r=.5):
    """
    Match feature spaces from different models by using kmeans.
    Hyperparameters and return are as defined in match_tensors_zipit.
    New feature space is just the cluster assignments, where num_clusters = merged_feature_dim
    """
    correlation = compute_correlation(metric["covariance"])
    correlation = correlation.clone()
    O = correlation.shape[0]
    remainder = int(O * (1-r))
    bound = O - remainder
    torch.diagonal(correlation)[:] = -10
    cluster_labels = KMeans(
        n_clusters=bound
    ).fit(-correlation.cpu().numpy()).labels_
    matches = torch.zeros((O, bound), device=correlation.device)
    for model_idx, match_idx in enumerate(cluster_labels):
        matches[model_idx, match_idx] = 1
    
    merge = matches / (matches.sum(dim=0, keepdim=True) + 1e-5)
    unmerge = matches
    return merge.T, unmerge


#####################################################################################################################################
############################################### DEBUGGING MATCHING/ALIGNMENT FUNCTIONS ##############################################
#####################################################################################################################################

def match_tensors_return_a(metric, r=.5, add_bias=False, **kwargs):
    """
    Matches feature spaces from different models by returning only the first.
    Mainly used for debugging purposes. 
    Hyperparameters and return are as defined in match_tensors_zipit.
    """
    correlation = metric["covariance"]
    O = int(correlation.shape[0] * (1-r))

    eye = torch.eye(O, device=correlation.device)
    unmerge = torch.cat([eye, 0*eye], dim=0)
    merge = unmerge / (unmerge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge


def match_tensors_randperm(metric, r=.5, **kwargs):
    """
    Matches feature spaces from different models by randomly permuting the 
    space of one onto another. 
    Mainly used for debugging.
    Hyperparameters and return are as defined in match_tensors_zipit.
    """
    correlation = metric["covariance"]
    O = int(correlation.shape[0] * (1-r))

    eye = torch.eye(O, device=correlation.device)[torch.randperm(O, device=correlation.device)]
    unmerge = torch.cat([eye, eye], dim=0)
    merge = unmerge / (unmerge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge

def match_tensors_destroy(metric, r=.5):
    """ Similar to match_tensors_randperm. """
    correlation = metric["covariance"]
    O = int(correlation.shape[0] * (1-r))

    eye = torch.eye(O, device=correlation.device)
    permd = eye[torch.randperm(O, device=correlation.device)]
    unmerge = torch.cat([eye, permd], dim=0)
    merge = unmerge / (unmerge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge