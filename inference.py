"""
LISN Inference & Evaluation Script
===================================
Loads trained weights, runs SR on test images, computes PSNR/SSIM.

Usage:
    python inference.py --weights /path/to/lisn_final.pth \
                        --test_hr /path/to/test/HR \
                        --test_lr /path/to/test/LR_x4 \
                        --output_dir /path/to/test/SR_x4
"""

import os
import sys
import time
import argparse
import glob

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lisn import build_lisn
from utils.metrics import compute_psnr_ssim


def sr_single_image(model, lr_path, device="cuda", scale=4):
    """Super-resolve a single grayscale image. Returns (sr_uint8, time_ms)."""
    lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
    if lr_img is None:
        raise FileNotFoundError(f"Cannot read: {lr_path}")

    lr_t = torch.from_numpy(lr_img.astype(np.float32) / 255.0)
    lr_t = lr_t.unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        if device == "cuda":
            _ = model(lr_t)  # warmup
            torch.cuda.synchronize()
        t0 = time.time()
        sr_t = model(lr_t)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000

    sr_np = sr_t.squeeze().cpu().numpy()
    sr_uint8 = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)
    return sr_uint8, elapsed_ms


def evaluate(
    weights: str,
    test_hr: str,
    test_lr: str,
    output_dir: str,
    scale: int = 4,
    embed_dim: int = 92,
    num_lisb: int = 6,
    device: str = "auto",
    results_csv: str = None,
):
    """Run SR inference on full test set and compute metrics."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = build_lisn(upscale=scale, embed_dim=embed_dim, num_lisb=num_lisb).to(device)
    ckpt = torch.load(weights, map_location=device)
    # Handle both raw state_dict and our checkpoint format
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights: {weights}")

    lr_files = sorted(
        glob.glob(f"{test_lr}/*.png") +
        glob.glob(f"{test_lr}/*.jpg") +
        glob.glob(f"{test_lr}/*.jpeg")
    )
    print(f"Processing {len(lr_files)} test images...\n")

    results = []
    for i, lr_path in enumerate(lr_files):
        fname = os.path.basename(lr_path)
        hr_path = os.path.join(test_hr, fname)
        if not os.path.exists(hr_path):
            continue

        sr_img, t_ms = sr_single_image(model, lr_path, device, scale)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)

        # Ensure same size
        if sr_img.shape != hr_img.shape:
            sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))

        psnr, ssim = compute_psnr_ssim(sr_img, hr_img, border=scale)
        cv2.imwrite(os.path.join(output_dir, fname), sr_img)

        results.append({"filename": fname, "psnr": psnr, "ssim": ssim, "time_ms": t_ms})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lr_files)} done...")

    df = pd.DataFrame(results)

    print(f"\n{'='*50}")
    print(f"  SR Evaluation (×{scale})")
    print(f"{'='*50}")
    print(f"  Images  : {len(df)}")
    print(f"  PSNR    : {df['psnr'].mean():.2f} dB")
    print(f"  SSIM    : {df['ssim'].mean():.4f}")
    print(f"  Avg time: {df['time_ms'].mean():.1f} ms")
    print(f"{'='*50}")

    if results_csv:
        df.to_csv(results_csv, index=False)
        print(f"  Saved: {results_csv}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LISN inference")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test_hr", required=True)
    parser.add_argument("--test_lr", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_lisb", type=int, default=8)
    parser.add_argument("--results_csv", default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
