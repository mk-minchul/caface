import os
import torch
from nets import tface_model
from nets import arcface_net


def build_model(model_name='ir_50'):
    if model_name == 'ir_101':
        return tface_model.IR_101(input_size=(112,112))
    elif model_name == 'ir_101_arcface':
        return arcface_net.iresnet100()
    else:
        raise ValueError('not a correct model name', model_name)


def load_pretrained(model, pretained_model_path):
    if pretained_model_path.startswith('pretrained_models'):
        # it means the path is relative to the project root
        project_root = os.getcwd().split('facerec_framework')[0] + 'facerec_framework'
        abs_path = os.path.join(project_root, pretained_model_path)
    else:
        abs_path = pretained_model_path
    checkpoint = torch.load(abs_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # from pytorch lightning checkpoint
        model_statedict = checkpoint['state_dict']
        renamed = {key[6:]: val for key, val in model_statedict.items() if key.startswith('model.')}
        toupdate_statedict = model.state_dict()
        assert len(renamed) == len(toupdate_statedict)
        len_orig = len(toupdate_statedict)
        toupdate_statedict.update(renamed)
        assert len(toupdate_statedict) == len_orig
        match_res = model.load_state_dict(toupdate_statedict, strict=False)

        # head can have missing keys
        assert len(match_res.unexpected_keys) == 0
        assert all([key.startswith('head') for key in match_res.missing_keys])
    else:
        model.load_state_dict(checkpoint)
    return model


def freeze_part(model, arch):
    if arch == 'ir_50':
        start_train_key = 'body.8.res_layer.0.weight'
    else:
        raise ValueError('detach not implemented yet')

    for name, param in model.named_parameters():
        if name == start_train_key:
            break
        param.requires_grad = False
