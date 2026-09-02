"""
Dataset preparation: generate HR/LR image pairs from FLIR ADAS thermal images.

Usage:
    python prepare_dataset.py --flir_root /path/to/FLIR_ADAS_1_3 --output_root /path/to/IR_SR

This creates:
    output_root/
      train/HR/        (8862 images from FLIR train split)
      train/LR_x4/
      test/HR/          (1366 images from FLIR val split)
      test/LR_x2/
      test/LR_x4/
"""

import os
import glob
import argparse
import cv2


def generate_pairs(source_dir, hr_dir, lr_dirs, scales, max_images=None):
    """Generate HR/LR pairs using bicubic downsampling.
    
    HR images are center-cropped so dimensions are divisible by max(scales).
    """
    os.makedirs(hr_dir, exist_ok=True)
    for d in lr_dirs.values():
        os.makedirs(d, exist_ok=True)

    images = sorted(
        glob.glob(f"{source_dir}/*.jpeg") +
        glob.glob(f"{source_dir}/*.jpg") +
        glob.glob(f"{source_dir}/*.png")
    )
    if max_images:
        images = images[:max_images]

    max_scale = max(scales)
    count = 0

    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        h_crop = h - (h % max_scale)
        w_crop = w - (w % max_scale)
        img = img[:h_crop, :w_crop]

        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        cv2.imwrite(os.path.join(hr_dir, filename), img)

        for s in scales:
            lr = cv2.resize(
                img, (w_crop // s, h_crop // s),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(os.path.join(lr_dirs[s], filename), lr)

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} images...")

    return count


def prepare_flir_dataset(flir_root, output_root):
    """Prepare full train + test sets from FLIR ADAS."""
    
    # --- Test set (from FLIR validation split) ---
    test_source = os.path.join(flir_root, "val", "thermal_8_bit")
    test_hr = os.path.join(output_root, "test", "HR")
    
    if os.path.isdir(test_hr) and len(os.listdir(test_hr)) > 0:
        print(f"Test HR already exists ({len(os.listdir(test_hr))} images). Skipping.")
    else:
        print("Generating test set HR/LR pairs...")
        n = generate_pairs(
            source_dir=test_source,
            hr_dir=test_hr,
            lr_dirs={
                2: os.path.join(output_root, "test", "LR_x2"),
                4: os.path.join(output_root, "test", "LR_x4"),
            },
            scales=[2, 4],
        )
        print(f"  Generated {n} test pairs.")

    # --- Training set (from FLIR train split) ---
    train_source = os.path.join(flir_root, "train", "thermal_8_bit")
    train_hr = os.path.join(output_root, "train", "HR")
    
    if os.path.isdir(train_hr) and len(os.listdir(train_hr)) > 0:
        print(f"Train HR already exists ({len(os.listdir(train_hr))} images). Skipping.")
    else:
        print("Generating training set HR/LR pairs...")
        n = generate_pairs(
            source_dir=train_source,
            hr_dir=train_hr,
            lr_dirs={
                4: os.path.join(output_root, "train", "LR_x4"),
            },
            scales=[4],
        )
        print(f"  Generated {n} training pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flir_root", required=True, help="Path to FLIR_ADAS_1_3/")
    parser.add_argument("--output_root", required=True, help="Output path for IR_SR/")
    args = parser.parse_args()
    prepare_flir_dataset(args.flir_root, args.output_root)
