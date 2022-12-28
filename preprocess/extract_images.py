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

from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from mxdataset import MXDataset
import numpy as np
import cv2

def tensor_to_numpy(tensor):
    # -1 to 1 tensor to 0-255
    arr = tensor.numpy().transpose(1,2,0)
    return ((arr * 0.5 + 0.5) * 255).astype(np.uint8)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='./path_to/webface4m_subset')
    parser.add_argument('--save_dir', type=str, default='./path_to/webface4m_subset_images')
    args = parser.parse_args()

    train_dataset = MXDataset(root_dir=args.dataset)
    dataloader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        imgs, tgts = batch
        count = {}
        for image, tgt in zip(imgs, tgts):
            label = str(tgt.item())
            image_uint8 = tensor_to_numpy(image)
            if label not in count:
                count[label] = []
            count[label].append(label)
            image_save_path = os.path.join(args.save_dir, str(tgt.item()), f'{len(count[label])}.jpg')
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            cv2.imwrite(image_save_path, image_uint8)


