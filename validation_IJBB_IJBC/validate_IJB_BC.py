import sys
import pyrootutils
import os
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))
from caface import trainer
from validation_IJBB_IJBC.insightface_ijb_helper.dataloader import prepare_dataloader
from validation_IJBB_IJBC.insightface_ijb_helper import eval_helper_fusion
from validation_IJBB_IJBC import fusion
from validation_IJBB_IJBC.fusion import cluster_and_aggregate
import os
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
from functools import partial


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def infer_images(model, aggregator, img_root, landmark_list_path, batch_size, use_flip_test):
    img_list = open(landmark_list_path)

    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_paths = []
    landmarks = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(img_root, name_lmk_score[0])
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_paths.append(img_path)
        landmarks.append(lmk)
        faceness_scores.append(name_lmk_score[-1])

    print('total images : {}'.format(len(img_paths)))
    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=0, image_size=(112,112))

    model.eval()
    features = []
    norms = []
    cinterms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):

            if hyper_param.intermediate_type == 'map':
                feature, norm, intermediate = model(images.to("cuda:0"), return_intermediate=True)
                compressed_intermediate = aggregator.compress_intermediate(intermediate)
            elif hyper_param.intermediate_type == 'style':
                feature, norm, intermediate = model(images.to("cuda:0"), return_style=[int(i) for i in hyper_param.style_index.split(',')])
                compressed_intermediate = aggregator.compress_intermediate(intermediate)
            else:
                raise ValueError('not a correct intermediate')

            if use_flip_test:
                raise ValueError('not implemented yet')
                fliped_images = torch.flip(images, dims=[3])
                if hyper_param.intermediate_type == 'map':
                    flipped_feature, flipped_norm, flipped_intermediate = model(fliped_images.to("cuda:0"), return_intermediate=True)
                    flipped_compressed_intermediate = aggregator.compress_intermediate(intermediate)
                elif hyper_param.intermediate_type == 'style':
                    flipped_feature, flipped_norm, flipped_intermediate = model(fliped_images.to("cuda:0"), return_style=[int(i) for i in hyper_param.style_index.split(',')])
                    flipped_compressed_intermediate = aggregator.compress_intermediate(intermediate)
                else:
                    raise ValueError('not a correct intermediate')

                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())
                cinterms.append(compressed_intermediate.cpu().numpy())
                features.append(flipped_feature.cpu().numpy())
                norms.append(flipped_norm.cpu().numpy())
                cinterms.append(flipped_compressed_intermediate.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())
                cinterms.append(compressed_intermediate.cpu().numpy())

    features = np.concatenate(features, axis=0)
    img_feats = np.array(features).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    norms = np.concatenate(norms, axis=0)
    cinterms = np.concatenate(cinterms, axis=0)

    assert len(features) == len(img_paths)

    return img_feats, faceness_scores, norms, cinterms


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do ijb test')
    # general
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ijb_meta_path', type=str, default='IJB/insightface_helper/ijb')
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--center_path', type=str, required=True)

    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--dataset_name', default='IJBB', type=str, help='dataset_name, set to IJBC or IJBB')
    parser.add_argument('--fusion_method', type=str, default='cluster_and_aggregate', choices=('average', 'cluster_and_aggregate'))
    parser.add_argument('--use_flip_test', type=str2bool, default='False')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    print('inference on {}'.format(args.pretrained_model_path))
    # load model
    ckpt = torch.load(args.pretrained_model_path)
    hyper_param = ckpt['hyper_parameters']
    args.task = hyper_param['task']
    args.task_path = args.task
    args.model_name = hyper_param['arch']
    args.prefix = hyper_param['prefix']

    hyper_param['start_from_model_statedict'] = ''
    hyper_param['data_root'] = args.data_root
    hyper_param['center_path'] = args.center_path
    hyper_param['style_index'] = ','.join([str(i) for i in hyper_param['style_index']])
    hyper_param = dotdict(hyper_param)

    trainer_mod = trainer.Trainer(**hyper_param)
    trainer_mod.load_state_dict(ckpt['state_dict'])
    model = trainer_mod.model
    aggregator = trainer_mod.aggregator
    model.to("cuda:0")
    aggregator.to('cuda:0')
    model.eval()
    aggregator.eval()

    run_name = os.path.basename(args.pretrained_model_path).split('.')[0]
    save_path = os.path.join(f'./{dataset_name}_result', run_name, f"fusion_{args.fusion_method}")
    if not args.use_flip_test:
        save_path += '_noflip'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path: {}'.format(save_path))

    use_flip_test = args.use_flip_test
    print('use_flip_test', use_flip_test)
    print('fusion_method', args.fusion_method)

    # # Step1: Load Meta Data
    templates, medias = eval_helper_fusion.read_template_media_list(
        os.path.join(args.data_root, args.ijb_meta_path,
                     f'{dataset_name}/meta', f'{dataset_name.lower()}_face_tid_mid.txt'))
    p1, p2, label = eval_helper_fusion.read_template_pair_list(
        os.path.join(args.data_root, args.ijb_meta_path,
                     f'{dataset_name}/meta', f'{dataset_name.lower()}_template_pair_label.txt'))

    # # Step 2: Get Image Features
    img_root = os.path.join(args.data_root, args.ijb_meta_path, f'{dataset_name}/loose_crop')
    landmark_list_path = os.path.join(args.data_root, args.ijb_meta_path,
                                      f'{dataset_name}/meta/{dataset_name.lower()}_name_5pts_score.txt')
    img_feats, faceness_scores, norms, cinterms = infer_images(model=model,
                                                               aggregator=aggregator,
                                                               img_root=img_root,
                                                               landmark_list_path=landmark_list_path,
                                                               batch_size=args.batch_size,
                                                               use_flip_test=use_flip_test)
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))


    img_feats = img_feats * norms

    result_dicts = []
    if args.fusion_method == 'cluster_and_aggregate':
        Ns = [4, 8, 32, 64, 128, 256, 9999]
        # 9999 is longer than any probe length. So it is not splitting.
    else:
        Ns = [9999]
    for N in Ns:
        print("N", N)
        if args.fusion_method == 'average':
            fuse_fn = fusion.average.average
        elif args.fusion_method == 'cluster_and_aggregate':
            fuse_fn = partial(cluster_and_aggregate.aggregator_fuse_ijbb,
                              aggregator=aggregator,
                              max_feature_num=N)
        else:
            raise ValueError('not a correct fusion meothd')

        with torch.no_grad():
            template_norm_feats, unique_templates = \
                eval_helper_fusion.image2template_feature_custom(img_feats,
                                                                 templates,
                                                                 medias,
                                                                 fuse_fn,
                                                                 force_avg_gallery=False,
                                                                 all_intermediate=torch.tensor(cinterms))
        score = eval_helper_fusion.verification(template_norm_feats, unique_templates, p1, p2)
        verification_result_str, result_dict = eval_helper_fusion.calc_tpr_fpr(score, label)
        result_dict['Split'] = 'N:{}'.format(N)
        print(result_dict)
        result_dicts.append(result_dict)
    result = pd.DataFrame(result_dicts)
    result.to_csv(os.path.join(save_path, 'result_by_split.csv'))
