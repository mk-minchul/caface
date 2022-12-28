import os
import torchvision.datasets as datasets

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd

import os
import sys
from dataset_helpers.insightface_ijb_helper import eval_helper_identification
from dataset_helpers.insightface_ijb_helper import eval_helper as eval_helper_verification



def make_face_ijbb_dataset(ijb_root, val_data_path, num_sample_per_identity, **kwargs):
    dataset_name = 'IJBB'
    landmark_list_path = os.path.join('{}/{}/meta/{}_name_5pts_score.txt'.format(ijb_root, dataset_name, dataset_name.lower()))

    img_list = open(landmark_list_path)
    files = img_list.readlines()
    print('IJBB files:', len(files))

    img_paths = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(val_data_path, name_lmk_score[0])
        img_paths.append(img_path)

    dataset = IJBDataset(img_filenames=img_paths)

    # specify what gallery images are for style calculation later
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(ijb_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    unique_templates = np.unique(templates)
    gallery_feature_index = []
    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        if count_template < 1845:
            gallery_feature_index.append(ind_t)
    gallery_feature_index = np.unique(np.concatenate(gallery_feature_index, axis=0))
    gallery_images_paths = [img_paths[i] for i in gallery_feature_index]
    dataset.gallery_images_paths = gallery_images_paths
    dataset.gallery_feature_index = gallery_feature_index

    return dataset

    
class IJBDataset(Dataset):
    def __init__(self, img_filenames):
        self.img_filenames = img_filenames

        self.mean_ = torch.FloatTensor([0.5, 0.5, 0.5])[:, None, None]
        self.std_ = torch.FloatTensor([0.5, 0.5, 0.5])[:, None, None]

    def __len__(self):
        return len(self.img_filenames)

    def load_image(self, fname, mode='RGB', return_orig=False):
        img = np.array(Image.open(fname).convert(mode))
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        if return_orig:
            return out_img, img
        else:
            return out_img

    def do_normalize_inputs(self, x):
        # rgb -> bgr 
        # 0-1 to -1-1
        x = torch.tensor(x)
        return (torch.flip(x, dims=[0]) - self.mean_.to(x.device)) / self.std_.to(x.device)

    def __getitem__(self, i):

        image = self.load_image(self.img_filenames[i], mode='RGB')
        image = self.do_normalize_inputs(image)
        # it is reading as bgr
        
        return i, image



def ijbb_evaluation(data_root, ijb_root, dataset_name, image_features, best_fn, all_intermediate=None):

    # read metadata
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(data_root, ijb_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    p1, p2, label = eval_helper_verification.read_template_pair_list(
        os.path.join(data_root, ijb_root, '%s/meta' % dataset_name, '%s_template_pair_label.txt' % dataset_name.lower()))

    # only do verification if you have full validation set inferred
    if len(templates) == len(image_features):
        # fusion
        template_norm_feats, unique_templates = \
                                    eval_helper_verification.image2template_feature_custom(image_features.numpy(), 
                                                                                            templates, 
                                                                                            medias,
                                                                                            best_fn,
                                                                                            all_intermediate=all_intermediate)
        score = eval_helper_verification.verification(template_norm_feats, unique_templates, p1, p2)
        verification_result_str, result_dict = eval_helper_verification.calc_tpr_fpr(score, label)
        print('verification_result')
        print(verification_result_str)

        return result_dict
    else:
        print('not enough ijbb data is inferred')
        return {}