#!/usr/bin/env python3
"""
seed_creation_fmri.py

Generate reproducible train/val/test splits for 4D fMRI preprocessed volumes only.

Features:
  - Reads all .pt files in a specified input directory
  - Filters only those with 4D shape (T, D, H, W)
  - Configurable train/validation/test ratios
  - Fixed random seed for reproducibility
  - Saves splits to JSON
  - Computes and prints age and sex statistics per split based on metadata CSV

Usage:
    python seed_creation_fmri.py \
        --input_dir /path/to/preprocessed_fmri_pt \
        --metadata_csv /path/to/adni_metadata.csv \
        --subject_column subject \
        --output_split splits_fmri.json \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 2025
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for 4D fMRI dataset"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing preprocessed .pt volumes'
    )
    parser.add_argument(
        '--metadata_csv', type=str, required=True,
        help='CSV file with subject metadata (age, sex, etc.)'
    )
    parser.add_argument(
        '--subject_column', type=str, default='subject',
        help='Column name in metadata CSV for subject ID'
    )
    parser.add_argument(
        '--output_split', type=str, default='splits_fmri.json',
        help='Path to save JSON splits file'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.7,
        help='Fraction of data to use for training'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.15,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # list all .pt files and filter 4D
    all_files = os.listdir(args.input_dir)
    pt_files = [f for f in all_files if f.endswith('.pt')]
    fmri_subjects = []
    for f in pt_files:
        path = os.path.join(args.input_dir, f)
        try:
            tensor = torch.load(path)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                fmri_subjects.append(os.path.splitext(f)[0])
        except Exception:
            continue

    if not fmri_subjects:
        raise RuntimeError(f"No 4D fMRI .pt files found in {args.input_dir}")

    subjects = np.array(fmri_subjects)
    np.random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_ids = subjects[:n_train].tolist()
    val_ids = subjects[n_train:n_train + n_val].tolist()
    test_ids = subjects[n_train + n_val:].tolist()

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    with open(args.output_split, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved split file to {args.output_split}")

    # compute metadata stats
    meta = pd.read_csv(args.metadata_csv)
    subj_col = args.subject_column
    if subj_col not in meta.columns:
        raise KeyError(f"Column '{subj_col}' not in {args.metadata_csv}")

    def describe(name, ids):
        subset = meta[meta[subj_col].astype(str).isin(ids)]
        print(f"\n{name} set: {len(ids)} subjects")
        if 'age' in subset.columns:
            print(f"  Age mean: {subset['age'].mean():.2f}, std: {subset['age'].std():.2f}")
        if 'sex' in subset.columns:
            counts = subset['sex'].value_counts()
            print(f"  Sex counts:\n{counts.to_string()}")

    describe('Train', train_ids)
    describe('Validation', val_ids)
    describe('Test', test_ids)

if __name__ == '__main__':
    main()
