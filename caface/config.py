import argparse
import sys
import os
from time import gmtime, strftime
import importlib


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_root', type=str, default='')
    parent_parser.add_argument('--train_data_path', type=str, default='faces_emore/imgs')
    parent_parser.add_argument('--ijb_meta_path', type=str, default='IJB/insightface_helper/ijb')
    parent_parser.add_argument('--ijb_aligned_imgs_path', type=str, default='IJB/aligned/IJBB')
    parent_parser.add_argument('--style_dataset_name', type=str, default='')

    parent_parser.add_argument('--prefix', type=str, default='default')
    parent_parser.add_argument('--wandb_tags', type=str, default='')
    parent_parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    parent_parser.add_argument('--tpus', type=int, default=0, help='how many tpus')
    parent_parser.add_argument('--distributed_backend', type=str, default='ddp', choices=('dp', 'ddp', 'ddp2'),)
    parent_parser.add_argument('--use_16bit', action='store_true', help='if true uses 16 bit precision')
    parent_parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parent_parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')
    parent_parser.add_argument('--batch_size', default=256, type=int,
                               help='mini-batch size (default: 256), this is the total '
                                    'batch size of all GPUs on the current node when '
                                    'using Data Parallel or Distributed Data Parallel')
    parent_parser.add_argument('--val_batch_size', default=128, type=int)

    parent_parser.add_argument('--lr',help='learning rate',default=1e-3, type=float)
    parent_parser.add_argument('--lr_milestones', default='8,12,14', type=str, help='epochs for reducing LR')
    parent_parser.add_argument('--lr_gamma', default=0.1, type=float, help='multiply when reducing LR')
    parent_parser.add_argument('--optimizer_type', default='sgd', type=str)
    parent_parser.add_argument('--lr_scheduler', type=str, default='step')

    parent_parser.add_argument('--num_workers', default=16, type=int)
    parent_parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true')
    parent_parser.add_argument('--evaluate', action='store_true', help='use with start_from_model_statedict')
    parent_parser.add_argument('--resume_from_checkpoint', type=str, default='')

    parent_parser.add_argument('--start_from_model_optim_statedict', type=str, default='')
    parent_parser.add_argument('--start_from_model_statedict', type=str, default='')
    parent_parser.add_argument('--center_path', type=str, default='')

    parser = add_task_arguments(parent_parser)
    args = parser.parse_args()

    trainer_module = importlib.import_module('trainer')
    data_module = importlib.import_module('data')

    args.lr_milestones = [int(x) for x in args.lr_milestones.split(',')]

    # set working dir
    current_time = strftime("%m-%d_0", gmtime())
    project_root = os.path.dirname(os.getcwd())
    args.output_dir = os.path.join(project_root, 'experiments', args.prefix + "_" + current_time)
    if os.path.isdir(args.output_dir):
        while True:
            cur_exp_number = int(args.output_dir[-2:].replace('_', ""))
            args.output_dir = args.output_dir[:-2] + "_{}".format(cur_exp_number+1)
            if not os.path.isdir(args.output_dir):
                break
    
    return args, trainer_module, data_module


def add_task_arguments(parser):
    parser.add_argument('--arch', default='ir_se_50')
    parser.add_argument('--freeze_part',action='store_true') # freeze part of the backbone
    parser.add_argument('--freeze_model',action='store_true') # freeze model
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_value', type=float, default=0.0) # default is no clip

    parser.add_argument('--save_all_models', action='store_true')
    parser.add_argument('--num_images_per_identity', type=int, default=10)

    parser.add_argument('--center_loss_lambda', type=float, default=0.0)
    parser.add_argument('--memory_loss_lambda', type=float, default=0.0)

    parser.add_argument('--limit_train_batches', type=float, default=1.0)

    parser.add_argument('--aggregator_name', default='style')  # stlye or map
    parser.add_argument('--intermediate_type', type=str, default='style') # style or map
    parser.add_argument('--style_index', type=str, default='3,5') # style or map
    parser.add_argument('--decoder_name', type=str, default='cat_default')

    parser.add_argument('--use_precompute_trainrec', type=str, default='')
    parser.add_argument('--same_aug_within_group_prob', type=float, default=1.0)
    parser.add_argument('--datafeed_scheme', type=str, default='dual_multi_v1', choices=['single', 'dual_multi_v1'])
    parser.add_argument('--img_aug_scheme', type=str, default='v2', choices=['v2', 'v3'])

    parser.add_argument('--use_memory', action='store_true')

    return parser