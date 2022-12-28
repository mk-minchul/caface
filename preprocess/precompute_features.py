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

import argparse
import torch
import mxnet as mx
import time
import traceback
import numpy as np
import image_augmenter
import queue
from caface import model as model_module


def list_image(root, exts=['.jpeg', '.jpg', '.png']):
    # returns iterator of
    # (index, relative_path, label)
    i = 0
    cat = {}
    for path, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        files.sort()
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                if path not in cat:
                    cat[path] = len(cat)
                yield (i, os.path.relpath(fpath, root), cat[path])
                i += 1


def write_list(path_out, image_list):
    # img_idx \t label \t path_to_image
    print('saving at {}'.format(path_out))
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)


def make_list(image_root, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_list = list(list_image(image_root))
    print(f'len image_list: {len(image_list)}')
    write_list(save_path, image_list)
    return save_path


def read_list(path_in):
    # Reads the .lst file and generates corresponding iterator.
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print('lst should have at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def extract_and_save_feature(i, item, q_out, augmenter, backbone, image_root):
    # item: [0, '0_0_0000039/0_573.jpg', 0.0]
    list_idx = item[0]
    list_rel_path = item[1]
    list_label = item[2]

    fullpath = os.path.join(image_root, list_rel_path)
    aug_samples = augmenter.make_aug_samples(fullpath, num_aug=16)

    feature, norm, intermediate = backbone(aug_samples.to("cuda:0"), return_style=[3,5])

    # reshape features
    mag_feature = feature * norm
    intermediates = torch.cat(intermediate, dim=1) # [[16, 128, 2], [16, 256, 2]] -> [16, 384, 2]
    intermediates = intermediates.view(intermediates.shape[0], -1) # 16, 768
    save_features = torch.cat([mag_feature, intermediates], dim=1) # 16, 512 + 768 (1280)
    save_features = save_features.view(-1)  # 20480
    save_features_np = save_features.detach().cpu().numpy()
    save_features_np_fp16 = save_features_np.astype(np.float16)

    header = mx.recordio.IRHeader(0, list_label, list_idx, 0)
    try:
        s = mx.recordio.pack(header, save_features_np_fp16.tobytes())
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_root', type=str, default='./path_to/webface4m_subset_images')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/AdaFaceWebFace4M.ckpt')
    parser.add_argument('--save_dir', type=str, default='./AdaFaceWebFace4M')
    args = parser.parse_args()

    augmenter = image_augmenter.MultipleAugmenter()
    backbone = model_module.build_model(model_name='ir_101')
    backbone = model_module.load_pretrained(backbone, args.pretrained_model_path)
    backbone.to("cuda:0")
    backbone.eval()

    fname_lst = os.path.join(args.save_dir, 'train.lst')
    fname_idx = os.path.join(args.save_dir, 'train.idx')
    fname_rec = os.path.join(args.save_dir, 'train.rec')
    make_list(args.image_root, fname_lst)
    image_list = read_list(fname_lst)
    q_out = queue.Queue()
    record = mx.recordio.MXIndexedRecordIO(fname_idx, fname_rec, 'w')
    cnt = 0
    pre_time = time.time()
    for i, item in enumerate(image_list):
        extract_and_save_feature(i, item, q_out, augmenter, backbone, args.image_root)
        if q_out.empty():
            continue
        _, s, _ = q_out.get()
        record.write_idx(item[0], s)
        if cnt % 1000 == 0:
            cur_time = time.time()
            print('time:', cur_time - pre_time, ' count:', cnt)
            pre_time = cur_time
        cnt += 1