import re
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def get_all_files(root, extension_list=['.jpg', '.png', '.jpeg']):
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def to_tensor(pil_image, device):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])(pil_image).unsqueeze(0).to(device)

class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list):
        super(ListDatasetWithIndex, self).__init__()

        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        # load in BGR format as AdaFace model requires
        img = cv2.imread(self.img_list[idx])
        if img is None:
            print(self.img_list[idx])
            raise ValueError(self.img_list[idx])
        img = img[:,:,:3]

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


def prepare_imagelist_dataloader(img_list, batch_size, num_workers=0):

    image_dataset = ListDatasetWithIndex(img_list)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers)
    return dataloader



