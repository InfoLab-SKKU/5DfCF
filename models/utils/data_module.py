#!/usr/bin/env python3
"""
4D fMRI Preprocessing Pipeline

This script processes 4D fMRI NIfTI files only:

Features:
  - Scans input directory for .nii/.nii.gz files
  - Loads 4D volumes (X, Y, Z, T)
  - Intensity normalization over whole 4D volume (z-score or min-max)
  - Optional N4 bias field correction per frame
  - Optional histogram equalization per frame
  - Optional Gaussian smoothing per frame
  - Center-crop or pad each 3D frame to target_shape
  - Optional simple flip augmentation
  - Multiprocessing with progress bar
  - Detailed logging
  - Per-subject metadata JSON
  - Summary CSV report
"""

import os
import sys
import json
import argparse
import logging
import csv
import time
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import LoadImage, N4BiasFieldCorrection
from monai.networks.layers import GaussianFilter
from skimage import exposure
from tqdm import tqdm

def setup_logging(log_file=None):
    log_format = "%(asctime)s %(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[])
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        logging.getLogger().addHandler(h)

def center_crop_or_pad(volume, target_shape):
    D, H, W = volume.shape
    output = np.zeros(target_shape, dtype=volume.dtype)
    # compute slices and destinations
    slices_src = []
    slices_dst = []
    for cur, tgt in zip((D, H, W), target_shape):
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices_src.append(slice(start, start + tgt))
            slices_dst.append(slice(0, tgt))
        else:
            pad_before = (tgt - cur) // 2
            slices_src.append(slice(0, cur))
            slices_dst.append(slice(pad_before, pad_before + cur))
    cropped = volume[slices_src[0], slices_src[1], slices_src[2]]
    output[slices_dst[0], slices_dst[1], slices_dst[2]] = cropped
    return output

def histogram_equalization(frame):
    mask = frame > 0
    eq = exposure.equalize_adapthist(frame, clip_limit=0.03)
    frame[mask] = eq[mask]
    return frame

def extract_metadata(header):
    meta = {}
    if hasattr(header, 'get'):
        a = header.get('affine')
        meta['affine'] = a.tolist() if a is not None else None
        meta['spatial_shape'] = header.get('spatial_shape')
    else:
        try:
            meta['shape'] = header.get_data_shape()
        except:
            meta['shape'] = None
    return meta

def preprocess_subject(filename, input_dir, output_dir,
                       scaling, bias_correct, equalize,
                       smooth_sigma, augment, target_shape):
    logger = logging.getLogger('preprocess_subject')
    start = time.time()
    subj = os.path.splitext(os.path.splitext(filename)[0])[0]
    in_path = os.path.join(input_dir, filename)
    out_tensor = os.path.join(output_dir, f"{subj}.pt")
    out_meta   = os.path.join(output_dir, f"{subj}_meta.json")
    status = {'subject': subj, 'status': 'skipped', 'error': '', 'duration': 0.0}

    try:
        img, header = LoadImage(image_only=False)(in_path)
        vol = img.numpy()
        if vol.ndim != 4:
            logger.warning(f"{subj}: not 4D, skipping")
            return status

        # transpose to (T, D, H, W)
        vol4d = np.transpose(vol, (3, 0, 1, 2))
        mask4d = vol4d > 0

        # normalization
        if scaling == 'z-norm':
            m, s = vol4d[mask4d].mean(), vol4d[mask4d].std()
            vol4d = (vol4d - m) / (s + 1e-8)
        else:
            mn, mx = vol4d[mask4d].min(), vol4d[mask4d].max()
            vol4d = (vol4d - mn) / (mx - mn + 1e-8)
        vol4d[~mask4d] = 0

        T = vol4d.shape[0]
        processed = np.zeros((T, *target_shape), dtype=np.float32)

        for t in range(T):
            frame = vol4d[t]
            frame_mask = mask4d[t]

            if bias_correct:
                tmp = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                frame = N4BiasFieldCorrection()(tmp).squeeze().numpy()

            frame[~frame_mask] = 0

            if equalize:
                frame = histogram_equalization(frame)

            frame = center_crop_or_pad(frame, target_shape)

            if smooth_sigma > 0:
                tmp = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float()
                frame = GaussianFilter(sigma=smooth_sigma)(tmp).squeeze().numpy()

            processed[t] = frame

        if augment:
            # flip along depth axis
            processed = processed[:, ::-1, :, :].copy()

        os.makedirs(output_dir, exist_ok=True)
        tensor = torch.from_numpy(processed).half()
        torch.save(tensor, out_tensor)

        meta = extract_metadata(header)
        meta.update({
            'scaling': scaling,
            'bias_correct': bias_correct,
            'equalize': equalize,
            'smooth_sigma': smooth_sigma,
            'augment': augment,
            'target_shape': target_shape
        })
        with open(out_meta, 'w') as f:
            json.dump(meta, f, indent=2)

        status['status'] = 'success'

    except Exception as e:
        logger.error(f"{subj} error: {e}")
        status['status'] = 'failed'
        status['error'] = str(e)
    finally:
        status['duration'] = round(time.time() - start, 3)
        return status

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="4D fMRI Preprocessing")
    parser.add_argument('--input_dir',  required=True, help='directory of 4D fMRI NIfTI')
    parser.add_argument('--output_dir', required=True, help='where to save .pt and metadata')
    parser.add_argument('--scaling', choices=['z-norm','minmax'], default='z-norm')
    parser.add_argument('--bias_correct', action='store_true')
    parser.add_argument('--equalize',    action='store_true')
    parser.add_argument('--smooth', type=float, default=0.0)
    parser.add_argument('--augment',    action='store_true')
    parser.add_argument('--target_shape', nargs=3, type=int, default=[96,96,96])
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--report',     default='summary.csv')
    parser.add_argument('--log_file',   default=None)
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger('main')
    logger.info(f"Starting with {args.workers} workers")

    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.endswith(('.nii','.nii.gz')))
    logger.info(f"Found {len(files)} volumes")

    func = partial(preprocess_subject,
                   input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   scaling=args.scaling,
                   bias_correct=args.bias_correct,
                   equalize=args.equalize,
                   smooth_sigma=args.smooth,
                   augment=args.augment,
                   target_shape=tuple(args.target_shape))

    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(func, files), total=len(files)))

    report_path = os.path.join(args.output_dir, args.report)
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['subject','status','error','duration'])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Done. Summary saved to {report_path}")
