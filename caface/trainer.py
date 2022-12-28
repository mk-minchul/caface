import torch
import torch.nn as nn
import numpy as np
import time
from dataset_helpers import ijb_dataset
import fusion
import trainer_base
import model
from losses import cossim
from nets import aggregator, fusion_net
from functools import partial
from utils import dist_utils


class Trainer(trainer_base.BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()

        # backbone face recognition model
        self.model = model.build_model(model_name=self.hparams.arch)

        # model responsible for CN and AGN
        fusion_model = fusion_net.build_fusion_net(model_name=self.hparams.decoder_name, hparams=self.hparams)
        if fusion_model is not None:
            intermediate_outdim = fusion_model.config['intermediate_outdim']
            normemb_dim = fusion_model.config['norm_dim']
        else:
            intermediate_outdim = None
            normemb_dim = 32

        # wrapper around SIM CN and AGN
        self.aggregator = aggregator.build_aggregator(model_name=self.hparams.aggregator_name,
                                                      fusion_net=fusion_model,
                                                      style_index=self.hparams.style_index,
                                                      input_maker_outdim=intermediate_outdim,
                                                      use_memory=self.hparams.use_memory,
                                                      normemb_dim=normemb_dim,
                                                      )
        num_params = sum([np.prod(p.size()) for p in self.aggregator.parameters()])
        print("num aggregator params", num_params)

        # load precomputed center for center loss
        center_dict = torch.load(self.hparams.center_path)
        center = center_dict['center']
        self.center = nn.Embedding(center.shape[0], center.shape[1])
        self.center.weight.data = center
        self.center.requires_grad = False
        self.center.weight.requires_grad = False
        self.center.weight.data.requires_grad = False

        # loss
        self.ignore_index = -100
        self.cossim_loss = cossim.cossim_loss

        self.maybe_load_saved()

        if self.hparams.freeze_part:
            model.freeze_part(self.model, self.hparams.arch)

        if self.hparams.freeze_model:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        # prepare fusion function
        self.fuse_fn = partial(fusion.aggregator_fuse_ijbb,
                               aggregator=self.aggregator, 
                               max_feature_num=9999, )

        self.hparams.style_index = [int(i) for i in self.hparams.style_index.split(',')]
        self.full_eval = True


    def train(self, mode: bool = True):
        print('train call')
        super().train(mode=mode)
        if self.hparams.freeze_model:
            self.model.eval()


    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            images, labels, extra_images, dummy_image = batch
            aug_image = None
        elif len(batch) == 5:
            images, labels, extra_images, dummy_image, aug_image = batch
        else:
            images, labels, extra_images = batch
            dummy_image, aug_image = None, None

        if self.hparams.lr_scheduler == 'step':
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        elif self.hparams.lr_scheduler == 'cosine':
            lr = self.trainer.lr_schedulers[0]['scheduler']._get_lr(self.trainer.global_step)[-1]
        else:
            raise ValueError('not implemetned yet')

        recog_info, recon_info = self.forward(images, labels, extra_images, batch_idx, dummy_image, aug_image)

        loss = 0

        # center loss
        if self.hparams.center_loss_lambda > 0:
            if 'gt' in recog_info:
                center = recog_info['gt']
            else:
                center = self.center(recog_info['labels'])
            agg_keys = list(filter(lambda x: 'aggregated' in x, recog_info.keys()))
            cos_loss = 0
            for agg_key in agg_keys:
                aggregated = recog_info[agg_key]
                if not isinstance(aggregated, list):
                    aggregated = [aggregated]
                for agg in aggregated:
                    cos_loss = cos_loss + (self.cossim_loss(agg, center, detach_y=True) / len(aggregated) / len(agg_keys))
            loss = loss + (cos_loss * self.hparams.center_loss_lambda)
            self.log('train/cos_loss', cos_loss, on_step=True, on_epoch=True, logger=True)


        if self.hparams.memory_loss_lambda > 0.0:
            memory_aggregated = recog_info['memory_aggregated']
            single_aggregated = recog_info['single_aggregated']
            mem_loss = self.cossim_loss(memory_aggregated, single_aggregated, detach_y=False) / len(single_aggregated)
            loss = loss + (mem_loss * self.hparams.memory_loss_lambda)
            self.log('train/memory_loss_lambda', mem_loss, on_step=True, on_epoch=True, logger=True)

        # logging
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss


    def forward(self, clean_images, orig_labels, extra_images, batch_idx, dummy_image=None, aug_image=None):

        folded_mag_embeddings, folded_intermediate = self.feature_extract(clean_images, extra_images)

        # split inference in aggregation into two
        if self.hparams.datafeed_scheme == 'single':
            agg_result_1 = self.aggregator(folded_features=folded_mag_embeddings, intermediate=folded_intermediate)
            agg_result_2 = {}

        elif self.hparams.datafeed_scheme == 'dual_multi_v1':
            B, T, C = folded_mag_embeddings.shape
            split_mag_emb = torch.split(folded_mag_embeddings, T // 2, dim=1)
            split_intermediate = torch.split(folded_intermediate, T // 2, dim=1)
            agg_result_1 = {"aggregated": [], "sigma2": [], "weights": [], 'mu': []}
            agg_result_2 = {"aggregated": [], "sigma2": [], "weights": [], 'mu': []}
            half_T = split_mag_emb[0].shape[1]
            for i in list(range(2, half_T+1)):
                if i % 2 == 1:
                    continue
                subset_index = torch.arange(i)
                sub_agg_result_1 = self.aggregator(folded_features=split_mag_emb[0][:, subset_index, :],
                                                   intermediate=split_intermediate[0][:, subset_index, :])
                sub_agg_result_2 = self.aggregator(folded_features=split_mag_emb[1][:, subset_index, :],
                                                   intermediate=split_intermediate[1][:, subset_index, :])

                agg_result_1['aggregated'].append(sub_agg_result_1['aggregated'])
                agg_result_2['aggregated'].append(sub_agg_result_2['aggregated'])
                agg_result_1['weights'].append(sub_agg_result_1['weights'])
                agg_result_2['weights'].append(sub_agg_result_2['weights'])
                if 'sigma2' in sub_agg_result_1:
                    agg_result_1['sigma2'].append(sub_agg_result_1['sigma2'])
                    agg_result_2['sigma2'].append(sub_agg_result_2['sigma2'])
                if 'mu' in sub_agg_result_1:
                    agg_result_1['mu'].append(sub_agg_result_1['mu'])
                    agg_result_2['mu'].append(sub_agg_result_2['mu'])


            if self.hparams.memory_loss_lambda > 0.0:
                assert self.hparams.use_memory
                self.aggregator.memory_count = 0
                _agg_result_1 = self.aggregator(folded_features=split_mag_emb[0], intermediate=split_intermediate[0])
                _agg_result_2 = self.aggregator(folded_features=split_mag_emb[1], intermediate=split_intermediate[1])
                self.aggregator.memory_count = -1
                self.aggregator.memory = None
                _agg_result_3 = self.aggregator(folded_features=folded_mag_embeddings, intermediate=folded_intermediate)
                memory_aggregated = _agg_result_2['aggregated']
                single_aggregated = _agg_result_3['aggregated']
            else:
                memory_aggregated = None
                single_aggregated = None

        else:
            raise ValueError('not a correct datafeed_scheme')

        # make return dict
        for key in ['aggregated']:
            assert key in agg_result_1
        recog_info = {'labels': orig_labels}
        recog_info.update({k+"_1":v for k, v in agg_result_1.items()})
        recog_info.update({k+"_2":v for k, v in agg_result_2.items()})

        if self.hparams.memory_loss_lambda > 0.0:
            assert self.hparams.use_memory
            assert self.hparams.datafeed_scheme == 'dual_multi_v1'
            recog_info['memory_aggregated'] = memory_aggregated
            recog_info['single_aggregated'] = single_aggregated

        recon_info = {}
        return recog_info, recon_info
    

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx):

        # ijb
        image_index, images = batch
        if self.hparams.intermediate_type == 'map':
            embeddings, norms, intermediate = self.model(images, return_intermediate=True)
        elif self.hparams.intermediate_type == 'style':
            embeddings, norms, intermediate = self.model(images, return_style=self.hparams.style_index)
        else:
            raise ValueError('not a correct intermediate type')

        compressed_intermediate = self.aggregator.compress_intermediate(intermediate)
        labels = torch.ones_like(norms)  # dummy
        dataname = torch.tensor(norms)  # dummy

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'compressed_intermediate': compressed_intermediate.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'output': embeddings,
                'norm': norms,
                'compressed_intermediate': compressed_intermediate.to('cpu'),
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):
        self.eval_ijbb_set(outputs, stage_name='val')

        if hasattr(self.aggregator, 'reset_memory'):
            self.aggregator.reset_memory()

    def test_step(self, batch, batch_idx):
        self.full_eval = True
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        self.eval_ijbb_set(outputs, stage_name='test')

        if hasattr(self.aggregator, 'reset_memory'):
            self.aggregator.reset_memory()
        return None


    def eval_ijbb_set(self, outputs, stage_name):

        if hasattr(self.aggregator, 'reset_memory'):
            self.aggregator.reset_memory()

        all_ijb_outputs = self.gather_outputs_v2(outputs, unique_index_name='image_index')
        all_face_features = all_ijb_outputs['output']  # torch.Size([227630, 512])
        all_face_norms = all_ijb_outputs['norm']  # torch.size([227630, 1])
        all_intermediate = all_ijb_outputs['compressed_intermediate']  # torch.size([227630, 512])

        # verification
        if self.trainer.is_global_zero:
            dataset_name = 'IJBB'
            data_root = self.hparams.data_root
            ijb_root = self.hparams.ijb_meta_path
            image_features = all_face_features * all_face_norms
            start_time = time.time()
            ijbb_result_dict = ijb_dataset.ijbb_evaluation(data_root, ijb_root, dataset_name, image_features, 
                                                           best_fn=self.fuse_fn, 
                                                           all_intermediate=all_intermediate)
            end_time = time.time()
            print('IJBB evaluation taken time: {} seconds'.format(end_time - start_time))
            ijbb_result_dict = [ijbb_result_dict]
        else:
            ijbb_result_dict = []
            pass

        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            ijbb_result_dict_list = []
            _ijbb_result_dict_list = dist_utils.all_gather(ijbb_result_dict)
            for _ijbb_result_dict in _ijbb_result_dict_list:
                ijbb_result_dict_list.extend(_ijbb_result_dict)
        else:
            ijbb_result_dict_list = ijbb_result_dict
        ijbb_result_dict = ijbb_result_dict_list[0]

        if ijbb_result_dict:
            self.log(f'ijbb_{stage_name}/0.0001', ijbb_result_dict['0.0001'], rank_zero_only=True)
            self.log(f'ijbb_{stage_name}/0.001', ijbb_result_dict['0.001'], rank_zero_only=True)
            self.log(f'ijbb_{stage_name}/1e-05', ijbb_result_dict['1e-05'], rank_zero_only=True)


