import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
import torch
import os

from utils import os_utils
from utils import dist_utils
from einops import rearrange, repeat
import numpy as np
from nets.aggregator import get_styledim


class BaseTrainer(LightningModule):
    def __init__(self, **kwargs):
        super(BaseTrainer, self).__init__()

        self.save_hyperparameters()  # sets self.hparams

    def maybe_load_saved(self):
        if (not self.hparams.start_from_model_optim_statedict and \
                not self.hparams.start_from_model_statedict):
            # no loading
            return None

        # load pretrained model weights
        if self.hparams.start_from_model_optim_statedict:
            checkpoint_path = self.hparams.start_from_model_optim_statedict
        else:
            checkpoint_path = self.hparams.start_from_model_statedict

        if checkpoint_path.startswith('experiments'):
            proj_root = os.getcwd().split(os.environ.get("PROJECT_NAME"))[0] + os.environ.get("PROJECT_NAME")
            checkpoint_path = os.path.join(proj_root, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # from pytorch lightning checkpoint
            statedict = checkpoint['state_dict']
            assert checkpoint['hyper_parameters']['arch'] == self.hparams.arch
            # match_res = self.load_state_dict(statedict, strict=True)
            if len(self.state_dict()) == len(statedict):
                self.load_state_dict(statedict)
                return None
            model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
            self.model.load_state_dict(model_statedict, strict=True)

            if hasattr(self, 'aggregator'):
                print('loading aggregator params')
                aggregator_statedict = {key[11:]: val for key, val in statedict.items() if
                                        key.startswith('aggregator.')}
                if aggregator_statedict:
                    self.aggregator.load_state_dict(aggregator_statedict, strict=True)

        else:
            # from released models, (no head)
            match_res = self.model.load_state_dict(checkpoint)

            # head can have missing keys
            assert len(match_res.unexpected_keys) == 0
            assert all([key.startswith('head') for key in match_res.missing_keys])

    def on_epoch_start(self):
        if self.trainer.is_global_zero and self.current_epoch == 0:
            # one time copy of project files
            code_dir = os.path.dirname(os.path.abspath(__file__))
            os_utils.copy_project_files(code_dir, self.hparams.output_dir)


    def forward(self, images, labels):
        return NotImplementedError()


    def gather_outputs_v2(self, outputs, unique_index_name='idx'):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = dist_utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        assert isinstance(outputs_list[0], dict)
        value_keys = [key for key in outputs_list[0].keys() if
                      key != unique_index_name]  # ex: ['aggregated_features', 'template_type', 'template_idx']
        gathered_tensor_dict = {}
        for value_key in value_keys:
            gathered_tensor = torch.cat([out[value_key] for out in outputs_list], axis=0).to('cpu')
            gathered_tensor_dict[value_key] = gathered_tensor

        gathered_idx = torch.cat([out[unique_index_name] for out in outputs_list], axis=0).to('cpu')
        unique_dict = {}
        for value_key, tensor in gathered_tensor_dict.items():
            for i, idx in enumerate(gathered_idx):
                if idx.item() not in unique_dict:
                    unique_dict[idx.item()] = {}
                unique_dict[idx.item()][value_key] = tensor[i]

        unique_keys = sorted(unique_dict.keys())

        all_gathered_outputs = {}
        for value_key in value_keys:
            all_gathered_outputs[value_key] = torch.stack([unique_dict[key][value_key] for key in unique_keys], axis=0)

        return all_gathered_outputs

    def configure_optimizers(self):

        bn_weight_decay = self.hparams.weight_decay if self.hparams.optimizer_type == 'adamw' else 0.0

        if self.hparams.freeze_model:
            model_paras_wo_bn, model_paras_only_bn = [], []
        else:
            model_paras_wo_bn, model_paras_only_bn = self.split_parameters(self.model)
        model_config = {'params': model_paras_wo_bn, }
        model_only_bn_config = {'params': model_paras_only_bn, 'weight_decay': bn_weight_decay, }

        agg_params_hasdecay, agg_params_nodecay = self.set_weight_decay(self.aggregator)
        agg_config = {'params': agg_params_hasdecay, }
        agg_only_bn_config = {'params': agg_params_nodecay, 'weight_decay': bn_weight_decay, }

        print('LR: {}'.format(self.hparams.lr))
        print('weight_decay: {}'.format(self.hparams.weight_decay))
        print('self.hparams.optimizer_type', self.hparams.optimizer_type)

        if self.hparams.optimizer_type == 'sgd':
            optimizer = optim.SGD([
                model_config, model_only_bn_config,
                agg_config, agg_only_bn_config,
            ],
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum)
        elif self.hparams.optimizer_type == 'adam':
            optimizer = optim.Adam([
                model_config, model_only_bn_config,
                agg_config, agg_only_bn_config,
            ],
                weight_decay=self.hparams.weight_decay,
                lr=self.hparams.lr)
        elif self.hparams.optimizer_type == 'adamw':
            optimizer = optim.AdamW([
                model_config, model_only_bn_config,
                agg_config, agg_only_bn_config,
            ],
                weight_decay=self.hparams.weight_decay,
                lr=self.hparams.lr,
                eps=1e-8,
                betas=(0.9, 0.999))
        else:
            raise ValueError('not a correct optimizer')

        if self.hparams.lr_scheduler == 'step':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.hparams.lr_milestones,
                                                 gamma=self.hparams.lr_gamma)
            interval = 'epoch'
        else:
            raise ValueError('not a correct lr scheudler')

        if self.hparams.start_from_model_optim_statedict:
            checkpoint_path = self.hparams.start_from_model_optim_statedict
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            optimizer_states = checkpoint['optimizer_states']
            optimizer_states[0]['param_groups'][0]['lr'] = self.hparams.lr
            optimizer_states[0]['param_groups'][1]['lr'] = self.hparams.lr
            optimizer.load_state_dict(optimizer_states[0])

        return [optimizer], [{"scheduler": scheduler, "interval": interval}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.hparams.lr_scheduler == 'step':
            scheduler.step()
        else:
            scheduler.step_update(self.trainer.global_step)

    def set_weight_decay(self, model, skip_list=(), skip_keywords=()):
        has_decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                has_decay.append(param)
        return has_decay, no_decay

    def split_parameters(self, module):
        if module is None:
            return [], []

        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay

    @staticmethod
    def cossim(x, y):
        x = x / torch.norm(x, 2, dim=2, keepdim=True)
        y = y / torch.norm(y, 2, dim=2, keepdim=True)
        print(x.shape)
        print(y.shape)
        return (x * y).sum(dim=-1)

    def parse_saved_feature(self, features):
        if features.shape[-1] == 25600:
            # shape: 32, 4, 25600
            folded_features, folded_intermediates = torch.split(torch.tensor(features), [512, 25088], dim=2)
            B, T, _ = folded_intermediates.shape
            folded_intermediates = folded_intermediates.view(B, T, 512, 7, 7)
            raveled_intermediate = rearrange(folded_intermediates, 'b t c h w-> (b t) c h w')
            return folded_features, raveled_intermediate
        else:
            # stylebased
            if 'style1357' in self.hparams.use_precompute_trainrec:
                tot_styledim = get_styledim('1,3,5,7')
                split_dims = [64, 128, 256, 512]
                split_index = [1, 3, 5, 7]
                style_index_dict = dict(zip(split_index, np.arange(len(split_index))))
            elif 'style35' in self.hparams.use_precompute_trainrec:
                tot_styledim = get_styledim('3,5')
                split_dims = [128, 256]
                split_index = [3, 5]
                style_index_dict = dict(zip(split_index, np.arange(len(split_index))))
            elif 'style3_space3':
                tot_styledim = get_styledim('3,4')
                split_dims = [128, 784]
                split_index = [3, 4]
                style_index_dict = dict(zip(split_index, np.arange(len(split_index))))
            else:
                raise ValueError('not implemented style yet')
            folded_features, folded_intermediates = torch.split(torch.tensor(features), [512, tot_styledim * 2], dim=2)
            B, T, _ = folded_intermediates.shape
            folded_intermediates = folded_intermediates.view(B, T, tot_styledim, 2)
            folded_intermediates_split = torch.split(folded_intermediates, split_dims,
                                                     dim=2)  # [[4, 128, 2], [4, 256, 2]] <- [4, 384, 2]

            # only select style vectors you want to use ( e.g 1,3,5,7 -> 3,5 )
            folded_intermediates_split = [folded_intermediates_split[style_index_dict[using_style]]
                                          for using_style in self.hparams.style_index]

            raveled_intermediates = [rearrange(split, 'b t c1 c2-> (b t) c1 c2') for split in
                                     folded_intermediates_split]
            return folded_features, raveled_intermediates

    def feature_extract(self, clean_images, extra_images):
        if clean_images.ndim == 4:
            clean_images = clean_images.unsqueeze(1)  # B, 1, C, H, W

        if clean_images.ndim == 5:  # image data
            # feature extract
            assert extra_images.ndim == 5
            folded_embeddings, folded_norms, folded_intermediate = self.ravel_forward(clean_images, extra_images)
            folded_mag_embeddings = folded_embeddings * folded_norms  # multiply norm

        elif clean_images.ndim == 3:  # feature data
            # it is from the feature extracted dataset
            assert self.hparams.use_precompute_trainrec
            assert extra_images.ndim == 3
            B = clean_images.shape[0]

            folded_features1, raveled_intermediates1 = self.parse_saved_feature(clean_images)
            folded_features2, raveled_intermediates2 = self.parse_saved_feature(extra_images)

            raveled_compressed_intermediate1 = self.aggregator.compress_intermediate(raveled_intermediates1)
            raveled_compressed_intermediate2 = self.aggregator.compress_intermediate(raveled_intermediates2)

            folded_intermediate1 = rearrange(raveled_compressed_intermediate1, '(b t) c-> b t c', b=B)
            folded_intermediate2 = rearrange(raveled_compressed_intermediate2, '(b t) c-> b t c', b=B)

            folded_mag_embeddings = torch.cat([folded_features1, folded_features2], dim=1)
            folded_intermediate = torch.cat([folded_intermediate1, folded_intermediate2], dim=1)
        else:
            raise ValueError('not a correct data strucutre')
        if not self.hparams.use_16bit:
            folded_mag_embeddings = folded_mag_embeddings.to(torch.float32)
            folded_intermediate = folded_intermediate.to(torch.float32)
        return folded_mag_embeddings, folded_intermediate

    def ravel_forward(self, images, extra_images):
        if images.ndim == 4:
            images = images.unsqueeze(1)

        stacked_images = torch.cat([images, extra_images], dim=1)

        B, T, C, H, W = stacked_images.shape
        raveled_images = rearrange(stacked_images, 'b t c h w -> (b t) c h w').contiguous()
        if self.hparams.intermediate_type == 'map':
            with torch.no_grad():
                raveled_embeddings, raveled_norms, raveled_intermediate = self.model(raveled_images,
                                                                                     return_intermediate=True)
            raveled_compressed_intermediate = self.aggregator.compress_intermediate(raveled_intermediate)
        elif self.hparams.intermediate_type == 'style':
            with torch.no_grad():
                raveled_embeddings, raveled_norms, raveled_intermediate = self.model(raveled_images,
                                                                                     return_style=self.hparams.style_index)
            raveled_compressed_intermediate = self.aggregator.compress_intermediate(raveled_intermediate)
        else:
            raise ValueError('not a correct intermediate type')

        folded_embeddings = rearrange(raveled_embeddings, '(b t) c -> b t c', t=T)
        folded_norms = rearrange(raveled_norms, '(b t) c -> b t c', t=T)
        folded_intermediate = rearrange(raveled_compressed_intermediate, '(b t) c -> b t c', t=T)

        return folded_embeddings, folded_norms, folded_intermediate


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
