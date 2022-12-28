import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from dataset_helpers import ijb_dataset
from face_datasets.feature_dataset import MultiAugMxFeatureDatasetV3

class DataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.train_data_path = kwargs['train_data_path']
        self.ijb_meta_path = kwargs['ijb_meta_path']
        self.ijb_aligned_imgs_path = kwargs['ijb_aligned_imgs_path']
        self.style_dataset_name = kwargs['style_dataset_name']

        self.batch_size = kwargs['batch_size']
        self.val_batch_size = kwargs['val_batch_size']
        self.data_root = kwargs['data_root']
        self.num_workers = kwargs['num_workers']
        self.num_workers = kwargs['num_workers']

        self.num_images_per_identity = kwargs['num_images_per_identity']

        self.use_precompute_trainrec = kwargs['use_precompute_trainrec']
        self.same_aug_within_group_prob = kwargs['same_aug_within_group_prob']

        self.datafeed_scheme = kwargs['datafeed_scheme']
        self.img_aug_scheme = kwargs['img_aug_scheme']

        print('[IMPORTANT] datafeed_scheme: {}'.format(self.datafeed_scheme))
        print('[IMPORTANT] img_aug_scheme: {}'.format(self.img_aug_scheme))



    def setup(self, stage=None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            print('creating train dataset')
            self.train_dataset = self.make_train_dataset()

            print('creating ijbb dataset')
            self.ijbb_dataset = ijb_dataset.make_face_ijbb_dataset(os.path.join(self.data_root, self.ijb_meta_path),
                                                                   os.path.join(self.data_root, self.ijb_aligned_imgs_path),
                                                                   num_sample_per_identity=1)

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.ijbb_dataset = ijb_dataset.make_face_ijbb_dataset(os.path.join(self.data_root, self.ijb_meta_path),
                                                                   os.path.join(self.data_root, self.ijb_aligned_imgs_path),
                                                                   num_sample_per_identity=1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        val_ijbb = DataLoader(self.ijbb_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)
        return [val_ijbb]

    def test_dataloader(self):
        test_ijbb = DataLoader(self.ijbb_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)
        return [test_ijbb]


    def make_train_dataset(self):

        if self.use_precompute_trainrec:
            if self.datafeed_scheme == 'dual_multi_v1':
                train_dataset = MultiAugMxFeatureDatasetV3(
                    root_dir=os.path.join(self.data_root, self.use_precompute_trainrec),
                    num_images_per_identity=self.num_images_per_identity,
                    same_aug_prob=self.same_aug_within_group_prob)
            else:
                raise ValueError('not a correct datafeed_scheme')
        else:
            raise ValueError('not a correct img aug scheme')

        return train_dataset





