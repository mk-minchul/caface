import numbers
import mxnet as mx
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import pandas as pd


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


class BaseMXDataset(Dataset):
    def __init__(self, root_dir, swap_color_order=False):
        super(BaseMXDataset, self).__init__()
        self.to_PIL = transforms.ToPILImage()
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        path_imglst = os.path.join(root_dir, 'train.lst')

        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # grad image index from the record and know how many images there are.
        # image index could be occasionally random order. like [4,3,1,2,0]
        s = self.record.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))
        print('self.imgidx length', len(self.imgidx))

        if os.path.isfile(path_imglst):
            self.record_info = pd.DataFrame(list(read_list(path_imglst)))
            self.insightface_trainrec = False
        else:
            self.insightface_trainrec = True
            # make one yourself
            record_info = []
            for idx in self.imgidx:
                s = self.record.read_idx(idx)
                header, _ = mx.recordio.unpack(s)
                label = header.label
                row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
                record_info.append(row)
            self.record_info = pd.DataFrame(record_info)

        self.swap_color_order = swap_color_order
        if self.swap_color_order:
            print('[INFO] Train data in swap_color_order')

    def read_sample(self, index):
        idx = self.imgidx[index]
        s = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()

        if self.swap_color_order or 'webface' in self.root_dir.lower() or self.insightface_trainrec:
            # swap rgb to bgr since image is in rgb for webface
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        else:
            sample = self.to_PIL(sample)
        return sample, label

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        # return len(self.imgidx)
        raise NotImplementedError()


class LabelConvertedMXFaceDataset(BaseMXDataset):

    def __init__(self,
                 root_dir,
                 swap_color_order=False,
                 rec_label_to_another_label=None
                 ):
        # rec_label_to_another_label: dictionary converting record label to another label like torch ImageFolderLabel
        super(LabelConvertedMXFaceDataset, self).__init__(root_dir=root_dir, swap_color_order=swap_color_order)
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


class MXDataset(LabelConvertedMXFaceDataset):
    def __init__(self,
                 root_dir,
                 swap_color_order=False,
                 ):
        super(MXDataset, self).__init__(root_dir,
                                        swap_color_order=swap_color_order,
                                        rec_label_to_another_label=None)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index, skip_augment=False):
        sample, target, record_label = self.read_sample(index)
        if self.transform is not None:
            sample = self.transform(sample)

        # # must check that this correctly saves image color channel (cv2 assumes BGR color channel)
        import cv2
        # cv2.imwrite('./temp.png', 255 * (0.5 * sample.transpose(0, 1).transpose(1, 2).numpy() + 0.5))

        return sample, target
