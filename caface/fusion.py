import torch
import numpy as np


def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def aggregator_fuse_ijbb(features, template_indexes, intermediates, aggregator, max_feature_num=8):

    tensor_feature = torch.tensor([features])  # B=1, T, C
    intermediates = intermediates.unsqueeze(0)  # B=1, T, C
    assert tensor_feature.ndim == 3
    assert intermediates.ndim == 3
    if tensor_feature.shape[1] > max_feature_num:
        # aggregated = aggregator(tensor_feature.cuda(), intermediates.cuda()).cpu()  # aggregated: 1 x C
        num_bins = int(np.ceil(tensor_feature.shape[1] / max_feature_num))
        splits = np.array_split(torch.arange(tensor_feature.shape[1]), num_bins)
        aggregated = []
        print('ijbb splitted aggregation total {} with max: {}'.format(tensor_feature.shape[1], max_feature_num))
        for k, split_index in enumerate(splits):
            print('split {} length {}'.format(k, len(split_index)))
            sub_feature = tensor_feature[:, split_index, :]
            sub_intermediate = intermediates[:, split_index, :]
            # sub_feature: 1 x max_feature_num x C
            agg_result = aggregator(sub_feature.cuda(), sub_intermediate.cuda())
            sub_aggregated = agg_result['aggregated'].cpu()  # aggregated: 1 x C
            aggregated.append(sub_aggregated)
        aggregated = torch.stack(aggregated, dim=0).mean(0)  # merged_features: 1, T, C
    else:
        agg_result = aggregator(tensor_feature.cuda(), intermediates.cuda())
        aggregated = agg_result['aggregated'].cpu()  # aggregated: 1 x C
    assert aggregated.ndim == 2
    assert aggregated.shape[0] == 1
    aggregated = aggregated[0]

    return aggregated



