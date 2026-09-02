"""
LISN Training Script
====================
Trains LISN on infrared HR/LR pairs with L1 + Sobel edge loss.

Usage (CLI):
    python train.py --train_hr /path/to/train/HR --train_lr /path/to/train/LR_x4 \
                    --checkpoint_dir /path/to/checkpoints --epochs 500

Usage (Colab):
    %run train.py --train_hr {PATHS['train_hr']} ...
    
    Or import and call train() directly.

Paper training config (Section 4.2):
    - Adam optimizer: β1=0.9, β2=0.999, ε=1e-8
    - Original paper: LR 2e-4, halved every 200 epochs, 6000 total
    - This version: CosineAnnealingWarmRestarts (T_0=250, min=1e-6)
      → Better for short runs (500 epochs). LR cycles smoothly,
        warm restarts prevent premature convergence.
    - Patch size: 64×64
    - Edge loss weight α=0.1
    - Gradient clipping: max_norm=1.0
"""

import os
import sys
import time
import argparse
import glob

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lisn import build_lisn
from models.losses import LISNLoss
from data.dataset import IRSuperResDataset


def train(
    train_hr: str,
    train_lr: str,
    checkpoint_dir: str,
    epochs: int = 500,
    batch_size: int = 8,
    lr: float = 2e-4,
    patch_size: int = 128,
    scale: int = 4,
    edge_weight: float = 0.1,
    embed_dim: int = 92,
    num_lisb: int = 6,
    save_every: int = 50,
    repeat: int = 4,
    device: str = "auto",
):
    """Train LISN model.
    
    Key changes from v1:
      - patch_size=128 (was 64): larger receptive field, better structure learning
      - batch_size=8 (was 16): compensates for 4x larger patches in GPU memory
      - repeat=4: each image yields 4 random crops per epoch (4x more training data)
      - Net effect: model sees 4x more diverse patches per epoch
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Model ---
    model = build_lisn(
        upscale=scale, embed_dim=embed_dim, num_lisb=num_lisb,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LISN | Params: {total_params:,} ({total_params/1e3:.1f}K) | Device: {device}")

    # --- Dataset ---
    train_dataset = IRSuperResDataset(
        hr_dir=train_hr, lr_dir=train_lr,
        patch_size=patch_size, scale=scale, augment=True, repeat=repeat,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(train_dataset)} images | Batch: {batch_size} | Patches: {patch_size}×{patch_size}")

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = LISNLoss(edge_weight=edge_weight).to(device)

    # CosineAnnealingWarmRestarts: LR cycles between lr and lr_min.
    # T_0=250 means full cosine cycle every 250 epochs.
    # For 500 epochs: 2 full warm-restart cycles (learns fast, then refines, then fast again).
    # For 1000 epochs: 4 cycles. Scales naturally with any epoch count.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=250, T_mult=1, eta_min=1e-6
    )

    # --- Resume from checkpoint ---
    start_epoch = 0
    ckpt_files = sorted(glob.glob(f"{checkpoint_dir}/lisn_epoch_*.pth"))
    if ckpt_files:
        latest = ckpt_files[-1]
        start_epoch = int(os.path.basename(latest).split("_")[-1].split(".")[0])
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:
            # Fast-forward scheduler to correct position
            for _ in range(start_epoch):
                scheduler.step()
        print(f"Resumed from epoch {start_epoch}: {os.path.basename(latest)}")

    # --- Training loop ---
    print(f"\nTraining epochs {start_epoch+1} → {epochs}")
    print(f"LR schedule: CosineAnnealingWarmRestarts (T_0=250, min_lr=1e-6)\n")
    model.train()

    best_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            print(
                f"  Epoch [{epoch+1:4d}/{epochs}] "
                f"Loss: {avg_loss:.6f}  Time: {elapsed:.1f}s  "
                f"LR: {current_lr:.2e}"
            )

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = f"{checkpoint_dir}/lisn_epoch_{epoch+1:04d}.pth"
            torch.save(ckpt, path)
            print(f"  → Saved: {os.path.basename(path)}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if (epoch + 1) >= 50:  # don't save noisy early epochs
                torch.save(
                    {"epoch": epoch + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                    f"{checkpoint_dir}/lisn_best.pth",
                )


    # Final save
    final_ckpt = {
        "epoch": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    final_path = f"{checkpoint_dir}/lisn_final.pth"
    torch.save(final_ckpt, final_path)
    print(f"\nTraining complete. Final: {final_path}")
    print(f"Best loss: {best_loss:.6f} (saved as lisn_best.pth)")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LISN")
    parser.add_argument("--train_hr", required=True)
    parser.add_argument("--train_lr", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--edge_weight", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=92)
    parser.add_argument("--num_lisb", type=int, default=6)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=4)
    args = parser.parse_args()
    train(**vars(args))
