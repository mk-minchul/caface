import sys, os
from nets.transformer import cluster_transformer

def build_fusion_net(model_name, hparams):
    if model_name == 'none':
        return None
    fusion_model = cluster_transformer.make_model(model_name, hparams)
    return fusion_model