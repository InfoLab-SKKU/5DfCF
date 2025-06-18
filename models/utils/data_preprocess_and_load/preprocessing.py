import os
import torch
import numpy as np
from monai.transforms import LoadImage
import torch.nn.functional as F

def center_crop_or_pad_3d(tensor, target_shape=(96, 96, 96)):
    """
    Center-crop or pad a 3D tensor (..., D, H, W) to target_shape.
    Leading dimensions (e.g., time/channels) are preserved.
    """
    *leading, D, H, W = tensor.shape
    pads = []
    slices = []
    for cur, tgt in zip((D, H, W), target_shape):
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
            pads.append((0, 0))
        else:
            pad_total = tgt - cur
            before = pad_total // 2
            after = pad_total - before
            slices.append(slice(0, cur))
            pads.append((before, after))
    # crop spatial dims
    cropped = tensor[..., slices[0], slices[1], slices[2]]
    # prepare pad config; F.pad expects reverse order
    pad_cfg = [p for dims in reversed(pads) for p in dims]
    # pad
    padded = F.pad(cropped, pad_cfg, mode="constant", value=0)
    return padded

def read_and_preprocess_4d(fpath, save_dir, scaling_method='z-norm', target_shape=(96, 96, 96)):
    """
    Load and preprocess a 4D fMRI NIfTI file:
      - LoadImage -> numpy array shape (X, Y, Z, T)
      - Transpose to (T, D, H, W)
      - Normalize (z-norm or min-max) over the whole volume
      - Spatial center-crop or pad each frame to target_shape
      - Save as float16 .pt with shape (T, 96, 96, 96)
    """
    subj = os.path.splitext(os.path.splitext(os.path.basename(fpath))[0])[0]
    print(f"Processing {subj} ...", flush=True)

    try:
        img, _ = LoadImage()(fpath)
    except Exception as e:
        print(f"  ❌ Failed to load {fpath}: {e}")
        return

    volume = img.numpy()
    if volume.ndim != 4:
        print(f"  ⚠️ Skipping {subj}, ndim={volume.ndim}")
        return

    # transpose to (T, D, H, W)
    vol4d = np.transpose(volume, (3, 0, 1, 2))
    mask = vol4d > 0

    # normalize
    if scaling_method == 'z-norm':
        mean = vol4d[mask].mean()
        std = vol4d[mask].std()
        vol4d = (vol4d - mean) / (std + 1e-8)
    else:  # 'minmax'
        mn, mx = vol4d[mask].min(), vol4d[mask].max()
        vol4d = (vol4d - mn) / (mx - mn + 1e-8)
    vol4d[~mask] = 0

    tensor4d = torch.from_numpy(vol4d).float()  # (T, D, H, W)
    # spatial crop/pad
    tensor4d = center_crop_or_pad_3d(tensor4d, target_shape=target_shape)
    # save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{subj}.pt")
    torch.save(tensor4d.type(torch.float16), out_path)
    print(f"  ✅ Saved: {out_path}")

def main():
    load_root = '/path/to/4d_fmri_nifti'      # directory with 4D fMRI files
    save_root = '/path/to/preprocessed_4d_pt' # directory to save preprocessed .pt
    scaling = 'z-norm'                        # 'z-norm' or 'minmax'
    target_shape = (96, 96, 96)               # spatial target shape

    # only .nii/.nii.gz files
    files = sorted([
        f for f in os.listdir(load_root)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    for fn in files:
        fpath = os.path.join(load_root, fn)
        read_and_preprocess_4d(
            fpath,
            save_root,
            scaling_method=scaling,
            target_shape=target_shape
        )

if __name__ == '__main__':
    main()
