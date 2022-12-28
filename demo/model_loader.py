import torch
from torch import nn
from omegaconf import OmegaConf
import sys
import pyrootutils
import os
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))
from caface.nets import fusion_net
from caface.nets import aggregator
from caface.model import build_model


class ModelWrapper(nn.Module):
    # dummy wrapper containing backbone and aggregator
    def __init__(self, fr_model, aggregator_model):
        super(ModelWrapper, self).__init__()
        self.model = fr_model
        self.aggregator = aggregator_model

def load_caface(ckpt_path, device='cuda:0'):

    print('Loading {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    hparam = OmegaConf.create(ckpt['hyper_parameters'])

    # load backbone face recognition model
    backbone = build_model(hparam['arch'])

    # fusion model: CN + AGN
    fusion_model = fusion_net.build_fusion_net(hparam['decoder_name'], hparam)
    # aggregator_model : SIM + CN + AGN
    aggregator_model = aggregator.build_aggregator(model_name=hparam.aggregator_name,
                                                          fusion_net=fusion_model,
                                                          style_index=','.join([str(i) for i in hparam.style_index]),
                                                          input_maker_outdim=fusion_model.config['intermediate_outdim'],
                                                          use_memory=hparam.use_memory,
                                                          normemb_dim=fusion_model.config['norm_dim'])

    # load pretrained weights
    model_wrapper = ModelWrapper(fr_model=backbone, aggregator_model=aggregator_model)
    result = model_wrapper.load_state_dict(ckpt['state_dict'], strict=False)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 1  # center.weight

    model = model_wrapper.model
    model.to(device)
    model.eval()

    aggregator_model = model_wrapper.aggregator
    aggregator_model.to(device)
    aggregator_model.eval()

    return aggregator_model, model, hparam
