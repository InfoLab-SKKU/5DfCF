#!/usr/bin/env python3
"""
metrics_fmri.py

Evaluation metrics for 4D fMRI data:
  - MSE, RMSE, MAE over entire 4D volume
  - Pearson correlation per voxel across time
  - SSIM averaged over all 3D frames
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Unsupported type: {type(x)}")


class Metrics4DFMRI:
    @staticmethod
    def mse(pred, target):
        """
        Mean squared error over 4D volumes.

        Args:
            pred, target: Tensor or ndarray of shape (T, D, H, W)
                          or (B, T, D, H, W)
        """
        p = _to_tensor(pred).float()
        t = _to_tensor(target).float()
        return torch.mean((p - t) ** 2).item()

    @staticmethod
    def rmse(pred, target):
        return float(torch.sqrt(torch.tensor(Metrics4DFMRI.mse(pred, target))))

    @staticmethod
    def mae(pred, target):
        p = _to_tensor(pred).float()
        t = _to_tensor(target).float()
        return torch.mean(torch.abs(p - t)).item()

    @staticmethod
    def pearson_corr(pred, target):
        """
        Pearson correlation coefficient averaged over voxels.

        Args:
            pred, target: Tensor or ndarray of shape (T, D, H, W)
                          or (B, T, D, H, W)
        """
        p = _to_tensor(pred).float()
        t = _to_tensor(target).float()

        if p.ndim == 5:
            B, T, D, H, W = p.shape
            p = p.view(B, T, -1)
            t = t.view(B, T, -1)
            corr_list = []
            for b in range(B):
                pv = p[b]
                tv = t[b]
                pv_mean = pv.mean(dim=0)
                tv_mean = tv.mean(dim=0)
                cov = ((pv - pv_mean) * (tv - tv_mean)).mean(dim=0)
                corr = cov / (pv.std(dim=0) * tv.std(dim=0) + 1e-8)
                corr_list.append(corr.mean().item())
            return float(np.mean(corr_list))

        if p.ndim == 4:
            T, D, H, W = p.shape
            p = p.view(T, -1)
            t = t.view(T, -1)
            pv_mean = p.mean(dim=0)
            tv_mean = t.mean(dim=0)
            cov = ((p - pv_mean) * (t - tv_mean)).mean(dim=0)
            corr = cov / (p.std(dim=0) * t.std(dim=0) + 1e-8)
            return corr.mean().item()

        raise ValueError("Input must be 4D or 5D tensor/ndarray")

    @staticmethod
    def ssim_time_series(pred, target, data_range=None):
        """
        Mean SSIM over all 3D volumes in a 4D series.

        Args:
            pred, target: Tensor or ndarray of shape (T, D, H, W)
            data_range: value range for SSIM calculation
        """
        p = _to_tensor(pred).cpu().numpy()
        t = _to_tensor(target).cpu().numpy()
        T = p.shape[0]
        scores = []
        for i in range(T):
            frame_p = p[i]
            frame_t = t[i]
            # average SSIM over axial slices
            slice_scores = []
            for z in range(frame_p.shape[0]):
                slice_scores.append(
                    ssim(
                        frame_p[z],
                        frame_t[z],
                        data_range=(data_range or (frame_t.max() - frame_t.min())),
                        gaussian_weights=True
                    )
                )
            scores.append(np.mean(slice_scores))
        return float(np.mean(scores))
