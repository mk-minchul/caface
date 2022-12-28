import numbers
import mxnet as mx
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import pandas as pd


class BaseMXFeatureDataset(Dataset):
    def __init__(self, root_dir):
        super(BaseMXFeatureDataset, self).__init__()
        self.to_PIL = transforms.ToPILImage()
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        path_imglst = os.path.join(root_dir, 'train.lst')

        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.record_info = pd.DataFrame(list(read_list(path_imglst)))
        s = self.record.read_idx(0)

        # grad image index from the record and know how many images there are.
        # image index could be occasionally random order. like [4,3,1,2,0]
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))
        print('self.imgidx length', len(self.imgidx))

    def read_sample(self, index):
        idx = self.imgidx[index]
        s = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        if '16' in self.root_dir:
            sample = np.frombuffer(img, dtype=np.float16)
        else:
            sample = np.frombuffer(img, dtype=np.float32)

        if len(sample) == 5 * 2432:
            # 5 aug + 1,3,5,7 style
            sample = sample.reshape(5, 2432)
            sample = sample.astype(np.float32)
        elif len(sample) == 6 * (512 + 1824):  # style3 spatial3
            sample = sample.reshape(6, 512 + 1824)
            sample = sample.astype(np.float32)
        elif len(sample) == 6 * (512 + 1920):  # 6 aug + 1,3,5,7
            sample = sample.reshape(6, 512 + 1920)
            sample = sample.astype(np.float32)
        elif len(sample) == 4 * 25600:
            # 4 aug with features
            sample = sample.reshape(4, 25600)
            sample = sample.astype(np.float32)
        elif len(sample) == 4 * 1280:
            # 4 aug with 3,5 style
            sample = sample.reshape(4, 1280)
            sample = sample.astype(np.float32)
        elif len(sample) == 16 * 1280:
            sample = sample.reshape(16, 1280)
            sample = sample.astype(np.float32)
        elif len(sample) == 16 * 2336:
            sample = sample.reshape(16, 2336)
            sample = sample.astype(np.float32)
        else:
            raise ValueError('not impleented feature')

        sample = torch.tensor(sample)
        # feature, intermediates = torch.split(torch.tensor(sample), [512, 768], dim=1)
        # intermediates = intermediates.view(4, 384, 2)

        return sample, label

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        # return len(self.imgidx)
        raise NotImplementedError()


class LabelConvertedMXFeatureDataset(BaseMXFeatureDataset):

    def __init__(self,
                 root_dir,
                 rec_label_to_another_label=None
                 ):
        # rec_label_to_another_label: dictionary converting record label to another label like torch ImageFolderLabel
        super(LabelConvertedMXFeatureDataset, self).__init__(root_dir=root_dir)
        if rec_label_to_another_label is None:
            # make one using path
            # image folder with 0/1.jpg

            # from record file label to folder name
            rec_label = self.record_info.label.tolist()
            foldernames = self.record_info.path.apply(lambda x: x.split('/')[0]).tolist()
            self.rec_to_folder = {}
            for i, j in zip(rec_label, foldernames):
                self.rec_to_folder[i] = j

            # from folder name to number as torch imagefolder
            foldernames = sorted(str(entry) for entry in self.rec_to_folder.values())
            self.folder_to_num = {cls_name: i for i, cls_name in enumerate(foldernames)}
            self.rec_label_to_another_label = {}

            # combine all
            for x in rec_label:
                self.rec_label_to_another_label[x] = self.folder_to_num[self.rec_to_folder[x]]


        else:
            self.rec_label_to_another_label = rec_label_to_another_label

    def __len__(self):
        return len(self.imgidx)

    def read_sample(self, index):
        sample, record_label = super().read_sample(index)
        new_label = self.rec_label_to_another_label[record_label.item()]
        new_label = torch.tensor(new_label, dtype=torch.long)
        return sample, new_label, record_label


class FeatureMXDataset(LabelConvertedMXFeatureDataset):
    def __init__(self,
                 root_dir,
                 rec_label_to_another_label=None,
                 ):
        super(FeatureMXDataset, self).__init__(root_dir, rec_label_to_another_label=rec_label_to_another_label)

    def __getitem__(self, index):
        aug_samples, target, record_label = self.read_sample(index)

        # orig_feature, crop_feature, photo_feature, blur_feature = aug_samples  4x1280
        select = np.random.randint(0, 4)
        sample = aug_samples[select]  # 1280

        return sample, target, record_label


class MultiAugMxFeatureDatasetV3(LabelConvertedMXFeatureDataset):
    def __init__(self,
                 root_dir,
                 rec_label_to_another_label=None,
                 num_images_per_identity=10,
                 same_aug_prob=1.0,
                 ):

        super(MultiAugMxFeatureDatasetV3, self).__init__(root_dir,
                                                         rec_label_to_another_label=rec_label_to_another_label)
        self.num_images_per_identity = num_images_per_identity
        self.same_aug_prob = same_aug_prob
        print('same_aug_prob: {}'.format(same_aug_prob))

    def get_augment_samples(self, sample, n):
        # aug_method = np.random.choice(['none', 'crop', 'photo', 'blur'])
        T = len(sample)
        aug_index = np.random.choice(T, n, replace=True)
        aug_samples = sample[aug_index]
        return aug_samples

    def __getitem__(self, index, return_path=False):

        sample, target, record_label = self.read_sample(index)
        same_label_images_info = self.record_info[self.record_info['label'] == record_label.item()]
        extra_sample = same_label_images_info.sample(self.num_images_per_identity - 1, replace=True)
        extra_index = extra_sample.index.tolist()

        # split into two groups
        perm_index = np.random.permutation(len(extra_index) + 1)
        all_index = np.array([index] + extra_index)[perm_index]
        half_point = len(all_index) // 2

        all_group_images = []
        for _ in range(2):
            num_iden = np.random.randint(1, max(half_point // 4, 3))
            iden_index = np.random.choice(all_index, num_iden)
            split_points = np.random.choice(half_point - 2, num_iden - 1, replace=False) + 1
            split_points.sort()
            split_points = split_points.tolist()
            count_per_iden = np.array(split_points + [half_point]) - np.array([0] + split_points)
            assert count_per_iden.sum() == half_point
            group_images = []
            for i_idx, count in zip(iden_index, count_per_iden):
                sample, _target, record_label = self.read_sample(i_idx)
                aug_images = torch.split(self.get_augment_samples(sample, n=count), 1)
                group_images.extend(aug_images)
            all_group_images.append(group_images)

        group1_samples = torch.cat([pil_img for pil_img in all_group_images[0]])
        group2_samples = torch.cat([pil_img for pil_img in all_group_images[1]])

        return group1_samples, target, group2_samples


def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    returns [idx, label, path]
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            assert line_len == 3
            item = {'idx': int(line[0]), "path": line[2], 'label': float(line[1])}
            yield item
