import torch
import numpy as np


def l2_normalize(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def infer_with(aggregator, features, interms, return_weights=False):
    if hasattr(aggregator, 'visualize_last'):
        version = 'v1'
        if return_weights:
            aggregator.visualize_last = True
    else:
        version = 'v2'

    if version == 'v1':
        aggregated = aggregator(features, interms, return_only_last=True)
        aggregated_np = aggregated.squeeze(0).detach().cpu().numpy()
        weights = None
        if hasattr(aggregator, 'weights'):
            weights = aggregator.weights
    else:
        agg_result = aggregator(features, interms)
        aggregated = agg_result['aggregated']
        aggregated_np = aggregated.squeeze(0).detach().cpu().numpy()
        if return_weights:
            weights = agg_result['weights']
        else:
            weights = None

    if return_weights:
        return aggregated_np, weights.cuda()
    else:
        return aggregated_np


def aggregator_fn(aggregator, features, intermediate, max_element, shuffle=False):
    if features.shape[1] > max_element:
        T = features.shape[1]
        num_split = int(np.ceil(T / max_element))
        if shuffle:
            splits = np.array_split(np.random.permutation(T), num_split)
        else:
            splits = np.array_split(np.arange(T), num_split)
        fused = None

        with torch.no_grad():
            for split_idx, split in enumerate(splits):
                subfeature = features[:,split,:]
                subinterm = intermediate[:,split, :]
                nan_index = subfeature.isnan().any(0).any(-1)
                subfeature = subfeature[:, ~nan_index, :]
                subinterm = subinterm[:, ~nan_index, :]
                if split_idx == 0:
                    aggregator.memory_count = 0
                fused = infer_with(aggregator, subfeature, subinterm, return_weights=False)
        fused_np = l2_normalize(fused, axis=0)
        aggregator.memory_count = 0
    else:
        aggregator.memory_count = 0
        fused = infer_with(aggregator, features, intermediate, return_weights=False)
        fused_np = l2_normalize(fused, axis=0)
        aggregator.memory_count = 0
    return fused_np


def aggregator_fuse_ijbb(features, template_indexes, intermediates, aggregator, max_feature_num=8):

    tensor_feature = torch.tensor([features]).cuda()  # B=1, T, C
    intermediates = intermediates.unsqueeze(0).cuda()  # B=1, T, C
    assert tensor_feature.ndim == 3
    assert intermediates.ndim == 3
    aggregated = aggregator_fn(aggregator, tensor_feature, intermediates, max_feature_num, shuffle=False)
    return aggregated

