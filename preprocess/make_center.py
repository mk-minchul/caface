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

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from caface import model as model_module
import torch
from mxdataset import MXDataset


def groupby_ops(value: torch.Tensor, labels: torch.LongTensor, op='sum') -> (torch.Tensor, torch.LongTensor):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=value.dtype).scatter_add_(0, labels, value)
    if op == 'mean':
        result = result / labels_count.float().unsqueeze(1)
    else:
        assert op == 'sum'
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels, labels_count


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/AdaFaceWebFace4M.ckpt')
    parser.add_argument('--dataset', type=str, default='./path_to/webface4m_subset')
    parser.add_argument('--save_dir', type=str, default='./AdaFaceWebFace4M')
    args = parser.parse_args()

    name = "center_{}_{}.pth".format(
        os.path.basename(args.pretrained_model_path).replace('.pth', '').replace('.ckpt', ''),
        os.path.basename(args.dataset).replace('.pth', ''))
    print('saving at')
    print(os.path.join(args.save_dir, name))

    # load model (This model assumes the input to be BGR image (cv2), not RGB (pil))
    model = model_module.build_model(model_name='ir_101')
    model = model_module.load_pretrained(model, args.pretrained_model_path)
    model.to("cuda:0")
    model.eval()

    with torch.no_grad():
        batch_size = 128
        train_dataset = MXDataset(root_dir=args.dataset)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        center = torch.zeros((len(train_dataset.record_info.label.unique()), 512))
        cul_count = torch.zeros(len(train_dataset.record_info.label.unique()))
        for batch in tqdm(dataloader):
            img, tgt = batch
            embedding, norm = model(img.cuda())
            sum_embedding, new_tgt, labels_count = groupby_ops(embedding.detach().cpu(), tgt, op='sum')
            for emb, tgt, count in zip(sum_embedding, new_tgt, labels_count):
                center[tgt] += emb
                cul_count[tgt] += count

        # flipped version
        train_dataset = MXDataset(root_dir=args.dataset)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        for batch in tqdm(dataloader):
            img, tgt = batch
            embedding, norm = model(img.cuda())
            sum_embedding, new_tgt, labels_count = groupby_ops(embedding.detach().cpu(), tgt, op='sum')
            for emb, tgt, count in zip(sum_embedding, new_tgt, labels_count):
                center[tgt] += emb
                cul_count[tgt] += count

    # normalize
    center = center / cul_count.unsqueeze(-1)
    center = center / torch.norm(center, 2, -1, keepdim=True)

    torch.save({'center': center, 'model': args.pretrained_model_path, 'dataset': args.dataset},
               os.path.join(args.save_dir, name))
