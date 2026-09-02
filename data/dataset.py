"""
Dataset classes for infrared image super-resolution.
Handles HR/LR pair loading, random cropping, and augmentation.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class IRSuperResDataset(Dataset):
    """Paired HR/LR infrared image dataset for training.
    
    Features:
      - Random patch cropping (default 128×128 on HR, 32×32 on LR for ×4)
      - Augmentation: random flips + 90° rotations (as in paper Section 4.2)
      - Repeat factor: each image yields multiple random crops per epoch
        (compensates for fewer epochs vs paper's 6000)
      - Grayscale, normalized to [0, 1]
    """
    def __init__(self, hr_dir, lr_dir, patch_size=128, scale=4, augment=True, repeat=4):
        self.hr_files = sorted(
            glob.glob(f"{hr_dir}/*.png") + 
            glob.glob(f"{hr_dir}/*.jpg") +
            glob.glob(f"{hr_dir}/*.jpeg")
        )
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.repeat = repeat  # each image sampled this many times per epoch

        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.hr_files) * self.repeat

    def __getitem__(self, idx):
        # Map repeated index back to actual file
        real_idx = idx % len(self.hr_files)
        hr_path = self.hr_files[real_idx]
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(self.lr_dir, filename)

        hr = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        lr = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Random crop (different random crop each time due to repeat)
        lr_patch = self.patch_size // self.scale
        lr_h, lr_w = lr.shape
        if lr_h > lr_patch and lr_w > lr_patch:
            y = np.random.randint(0, lr_h - lr_patch)
            x = np.random.randint(0, lr_w - lr_patch)
            lr = lr[y : y + lr_patch, x : x + lr_patch]
            hr = hr[
                y * self.scale : (y + lr_patch) * self.scale,
                x * self.scale : (x + lr_patch) * self.scale,
            ]

        # Augmentation (paper Section 4.2: random 90/180/270° + horizontal flip)
        if self.augment:
            if np.random.random() > 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
            if np.random.random() > 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
            k = np.random.randint(0, 4)
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()

        # (H, W) → (1, H, W)
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)
        return lr_tensor, hr_tensor


class IRTestDataset(Dataset):
    """Test dataset — loads full images without cropping or augmentation."""
    def __init__(self, hr_dir, lr_dir):
        self.hr_files = sorted(
            glob.glob(f"{hr_dir}/*.png") +
            glob.glob(f"{hr_dir}/*.jpg") +
            glob.glob(f"{hr_dir}/*.jpeg")
        )
        self.lr_dir = lr_dir

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(self.lr_dir, filename)

        hr = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        lr = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)
        return lr_tensor, hr_tensor, filename
