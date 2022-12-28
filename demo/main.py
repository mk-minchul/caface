import sys
import pyrootutils
import os
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))

import argparse
import cv2
from face_detection.detector import FaceDetector
from face_alignment.aligner import FaceAligner
from model_loader import load_caface
from dataset import get_all_files, natural_sort, prepare_imagelist_dataloader, to_tensor
from tqdm import tqdm
import numpy as np
from inference import infer_features, fuse_feature
import visualization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--ckpt_path", type=str, default='../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt')
    parser.add_argument("--video_path", type=str, default='examples/example1/probe.mp4')
    parser.add_argument("--save_root", type=str, default='result/examples1')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--fusion_method", type=str,
                        default='cluster_and_aggregate',
                        choices=['cluster_and_aggregate, average'])

    args = parser.parse_args()

    # load face detector and aligner
    detector = FaceDetector()
    aligner = FaceAligner()

    # load caface
    aggregator, model, hyper_param = load_caface(args.ckpt_path, device=args.device)

    # make save_dir
    save_dir = os.path.join(args.save_root, os.path.basename(args.video_path))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'detected'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'aligned'), exist_ok=True)

    # extract facial images from the probe video
    assert os.path.isfile(args.video_path)
    video = cv2.VideoCapture(args.video_path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(video_length), total=video_length):
        ret, frame = video.read()
        detected = detector.detect(frame)
        aligned = aligner.align(detected)
        if detected is not None:
            detected.save(os.path.join(save_dir, 'detected', f'{frame_num}.jpg'))
        if aligned is not None:
            aligned.save(os.path.join(save_dir, 'aligned', f'{frame_num}.jpg'))

    # get list of images extracted (using aligned in favor of detected, but alignment could fail sometimes)
    detected_images = natural_sort(get_all_files(os.path.join(save_dir, 'detected')))
    aligned_images = natural_sort(get_all_files(os.path.join(save_dir, 'aligned')))
    probe_image_list = [path.replace('detected', 'aligned') if path.replace('detected', 'aligned') else path
                        for path in detected_images]
    dataloader = prepare_imagelist_dataloader(probe_image_list, batch_size=16, num_workers=0)

    # infer singe image features
    probe_features, probe_intermediates = infer_features(dataloader, model, aggregator, hyper_param, device=args.device)
    # fuse features
    probe_fused_feature, probe_weights = fuse_feature(probe_features, aggregator, probe_intermediates,
                                                      method=args.fusion_method, device=args.device)

    # infer gallery for comparison with probe video
    gallery_path = os.path.join(os.path.dirname(args.video_path), 'gallery.jpg')
    if os.path.isfile(gallery_path):
        # infer gallery feature
        gallery_image = aligner.align(detector.detect(cv2.imread(gallery_path)))
        gallery_image_tensor = to_tensor(gallery_image, device=args.device)
        with torch.no_grad():
            gallery_feature, _ = model(gallery_image_tensor)
        gallery_feature = gallery_feature.detach().cpu().numpy()

        # make cosine similarity plot
        visualization.make_similarity_plot(os.path.join(save_dir, f'{args.fusion_method}.pdf'),
                                           probe_features, probe_weights, probe_fused_feature, probe_image_list,
                                           gallery_feature, gallery_image)
