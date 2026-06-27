"""
preprocessing.py
----------------
BraTS 2020 preprocessing pipeline.

Steps:
    1. Load NIfTI volumes (T1ce, T2, FLAIR) — T1 excluded
    2. Crop to 128x128x128 around brain centre
    3. Per-volume z-score normalisation on brain voxels only
    4. Stack modalities along channel dimension
    5. Remap labels {0,1,2,4} -> {0,1,2,3}

Usage:
    python preprocessing.py \
        --data_root  /path/to/BraTS2020_TrainingData \
        --output_root /path/to/preprocessed
"""

import os
import argparse
import numpy as np
import nibabel as nib


MODALITIES   = ["t1ce", "t2", "flair"]
TARGET_SHAPE = (128, 128, 128)


def load_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def crop_centre(volume: np.ndarray,
                target: tuple = TARGET_SHAPE) -> np.ndarray:
    d, h, w    = volume.shape
    td, th, tw = target
    pad = [
        (max(0, (td - d + 1) // 2), max(0, (td - d + 1) // 2)),
        (max(0, (th - h + 1) // 2), max(0, (th - h + 1) // 2)),
        (max(0, (tw - w + 1) // 2), max(0, (tw - w + 1) // 2)),
    ]
    volume = np.pad(volume, pad, mode="constant", constant_values=0)
    d, h, w = volume.shape
    sd, sh, sw = (d-td)//2, (h-th)//2, (w-tw)//2
    return volume[sd:sd+td, sh:sh+th, sw:sw+tw]


def normalise_volume(volume: np.ndarray) -> np.ndarray:
    """
    Per-volume z-score normalisation using brain (non-zero) voxels only.
    Background voxels remain zero after normalisation.
    """
    flat   = volume.reshape(-1)
    mask   = flat != 0
    result = np.zeros_like(flat, dtype=np.float32)
    if mask.sum() > 0:
        brain  = flat[mask]
        mean   = brain.mean()
        std    = brain.std()
        if std > 0:
            result[mask] = (brain - mean) / std
    return result.reshape(volume.shape)


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """
    BraTS 2020 raw labels: {0, 1, 2, 4}
    Remap 4 -> 3 for contiguous classes: {0, 1, 2, 3}
    """
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg.astype(np.int64)


def preprocess_case(case_dir: str) -> dict:
    case_id  = os.path.basename(case_dir)
    channels = []
    for mod in MODALITIES:
        vol = load_volume(os.path.join(case_dir, f"{case_id}_{mod}.nii.gz"))
        vol = crop_centre(vol)
        vol = normalise_volume(vol)
        channels.append(vol)
    image    = np.stack(channels, axis=0).astype(np.float32)
    seg      = load_volume(os.path.join(case_dir, f"{case_id}_seg.nii.gz"))
    seg      = crop_centre(seg)
    seg      = remap_labels(seg)
    return {"image": image, "label": seg}


def preprocess_dataset(data_root: str, output_root: str) -> None:
    os.makedirs(output_root, exist_ok=True)
    case_dirs = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    print(f"Found {len(case_dirs)} cases.")
    for idx, case_dir in enumerate(case_dirs):
        out_path = os.path.join(output_root,
                                f"{os.path.basename(case_dir)}.npz")
        if os.path.exists(out_path):
            continue
        try:
            data = preprocess_case(case_dir)
            np.savez_compressed(out_path,
                                image=data["image"],
                                label=data["label"])
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(case_dirs)}] saved.")
        except FileNotFoundError as e:
            print(f"  [WARNING] Skipping: {e}")
    print("Preprocessing complete.")


try:
    import torch
    from torch.utils.data import Dataset

    class BraTSDataset(Dataset):
        """PyTorch Dataset for preprocessed BraTS 2020 .npz files."""

        def __init__(self, npz_dir: str):
            self.files = sorted([
                os.path.join(npz_dir, f)
                for f in os.listdir(npz_dir)
                if f.endswith(".npz")
            ])

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            data = np.load(self.files[idx])
            return (torch.from_numpy(data["image"]),
                    torch.from_numpy(data["label"]))

except ImportError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    preprocess_dataset(args.data_root, args.output_root)
