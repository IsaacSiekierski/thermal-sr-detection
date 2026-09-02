# Thermal Image Super-Resolution and Its Effect on Object Detection

Does super-resolution on low-resolution thermal images actually help an object detector — or
does it only improve pixel-similarity metrics such as PSNR?

M.Sc. final project, Jerusalem College of Technology (Lev Academic Center), Data Mining track.
Author: Isaac Siekierski. Supervisor: Dr. Eyal Ben-Isaac.

---

## Summary

A lightweight super-resolution (SR) network was implemented from the description in
**"Infrared Image Super-Resolution via Lightweight Information Split Network"** (Liu et al.,
ICIC 2024) and trained on the thermal driving images of **FLIR ADAS 1.3**. It was then compared
against two published SR models — **SPAN** (NTIRE 2024 efficiency winner) and **DifIISR**
(CVPR 2025 task-oriented diffusion) — under a single frozen evaluation protocol.

Headline result on 1,366 held-out test images, frozen COCO-pretrained YOLOv8n:

| source | mAP@[.5:.95] | F1 [95% CI] | PSNR | LPIPS ↓ | params | ms/img |
|---|---|---|---|---|---|---|
| bicubic (floor) | 0.076 | 0.219 [0.208–0.231] | 28.00 | 0.532 | — | — |
| **this work** | **0.107** | **0.322 [0.310–0.333]** | **28.60** | **0.259** | **658,576** | **36.7** |
| SPAN | 0.073 | 0.237 [0.225–0.249] | 25.00 | 0.500 | 2,236,728 | 16.3 |
| HR (ceiling) | 0.226 | 0.639 [0.629–0.649] | — | — | — | — |

The 658K-parameter model trained on thermal data recovers **20.5%** of the detection gap caused
by the resolution loss. A 3.4× larger generic model recovers **−2.3%**, i.e. it hurts. The
confidence intervals do not overlap, and the ranking is reproduced by a second detector trained
on FLIR thermal images.

**Reproducibility note.** The code released with the LISN paper
(github.com/sad192/LISN-Infrared-Image-SR) builds Swin Transformer residual blocks rather than
the LISB blocks the paper describes: `models/network_hybrid.py` imports `SRB` from
`models/basicblock.py` and `SwinT` from `models/SwinT.py`, and declares `num_heads=8`. It also
imports `models/fusion.py`, which is not present in the repository. No pretrained weights are
available (open issue #4, 11 July 2025).
The architecture in `models/lisn.py` is an independent implementation written from the paper's
description; at the paper's default configuration (6 blocks, width 92) it counts 284,158
parameters against the 279K the paper reports. The trained model in this repository uses width
128 and 8 blocks.

---

## Repository layout

```
models/lisn.py                    SR architecture (SFE, LISB blocks, DFF, IIR)
models/losses.py                  L1 + Sobel edge loss
data/dataset.py                   paired HR/LR dataset, patches and augmentation
utils/prepare_dataset.py          FLIR -> HR/LR pair generation
utils/metrics.py                  grayscale PSNR / SSIM with border crop
train.py                          training entry point
inference.py                      inference and metrics entry point

notebooks/01_train_lisb_sr.ipynb              training run (Colab)
notebooks/02_model_comparison.ipynb        MAIN: full evaluation and comparison
notebooks/03_train_thermal_detector.ipynb  control detector training

checkpoints/lisb_sr_128x8_best.pth         trained SR model, 658,576 params
checkpoints/yolov8n_flir_thermal.pt        control detector

results/results_comparison.json            detection and fidelity metrics, gates, tables
results/results.json                       checkpoint provenance and convergence sweep
results/tableE_*.csv                       measured results
results/tableL_literature.csv              published figures, context only
```

---

## Requirements

Python 3.10+ and a CUDA GPU. Developed and run on Google Colab with a Tesla T4.

```bash
pip install -r requirements.txt
```

---

## Dataset

FLIR ADAS 1.3 is public. Download it from Kaggle
(`deepnewbie/flir-thermal-images-dataset`) and place it so that these paths exist:

```
<DRIVE_ROOT>/datasets/flir_adas/FLIR_ADAS_1_3/train/thermal_8_bit/
<DRIVE_ROOT>/datasets/flir_adas/FLIR_ADAS_1_3/train/thermal_annotations.json
<DRIVE_ROOT>/datasets/flir_adas/FLIR_ADAS_1_3/val/thermal_8_bit/
<DRIVE_ROOT>/datasets/flir_adas/FLIR_ADAS_1_3/val/thermal_annotations.json
```

Use `train/thermal_8_bit`, never `train/Annotated_thermal_8_bit` — the latter has the
ground-truth boxes drawn onto the pixels, and a detector trained on it learns to detect
rectangles.

---

## Running

### 1. Build the HR/LR pairs

```bash
python utils/prepare_dataset.py
```

Produces `datasets/IR_SR/train/{HR,LR_x4}` (8,862 pairs) and
`datasets/IR_SR/test/{HR,LR_x4}` (1,366 pairs). Each HR image is cropped to a multiple of 4
and downscaled ×4 with bicubic interpolation to produce its LR counterpart. Both are saved
as lossless PNG.

### 2. Train the SR model (optional — a trained checkpoint is included)

```bash
python train.py \
  --train_hr datasets/IR_SR/train/HR \
  --train_lr datasets/IR_SR/train/LR_x4 \
  --checkpoint_dir checkpoints \
  --epochs 300 --batch_size 8 --patch_size 128 \
  --embed_dim 128 --num_lisb 8 --scale 4
```

About 15 hours on a T4. Training resumes automatically from the newest checkpoint in
`checkpoint_dir`, so an interrupted session can simply be restarted.

### 3. Reproduce the evaluation

Open `notebooks/02_model_comparison.ipynb` in Colab and run the cells in order. It acquires
SPAN and DifIISR, runs all models on the same images, computes PSNR / SSIM / LPIPS,
runs YOLOv8n on every input source, computes mAP via COCOeval and F1 with bootstrap
confidence intervals, evaluates the six pre-declared gates, and writes
`results_comparison.json`.

Approximately 15–25 minutes with a warm cache; longer on the first run.

### 4. Control detector (optional)

`notebooks/03_train_thermal_detector.ipynb` trains YOLOv8n on the FLIR training split
(50 epochs, ~1.7 h on a T4) and writes `checkpoints/yolov8n_flir_thermal.pt`, which
Section 11 of the comparison notebook uses to verify that the model ranking does not
depend on the detector.

---

## Evaluation protocol

- **Fidelity:** grayscale PSNR and SSIM, 4-pixel border crop before comparison.
- **Perceptual:** LPIPS (AlexNet backbone), grayscale replicated to 3 channels.
- **Task:** frozen COCO-pretrained YOLOv8n, identical 640 letterbox for every source,
  predictions at confidence 0.001 so the full precision-recall curve is available,
  mAP@[.5:.95] via `pycocotools`, F1 at confidence 0.25 / IoU 0.5 with greedy class-aware
  matching, and 1,000-resample image-level bootstrap confidence intervals.
- **Gap closure:** `(metric(SR) − metric(bicubic)) / (metric(HR) − metric(bicubic))`.
- **Control:** the whole comparison repeated with a detector trained on FLIR thermal images.

External models are run with the weights their authors published, without any fine-tuning.
That contrast — domain-matched training versus off-the-shelf generality — is the variable
under study.

---

## Citation of the source work

Liu, S., Yan, K., Qin, F., Wang, C., Ge, R., Zhang, K., Huang, J., Peng, Y., Cao, J.
*Infrared Image Super-Resolution via Lightweight Information Split Network.* ICIC 2024.
arXiv:2405.10561

## Model naming

The SR model is named **LISB-SR-128/8** after the block it implements (Lightweight Information
Split Block), at width 128 with 8 blocks. Earlier drafts of this work called it `Light-SwinIR`;
that name was wrong and has been retired. The trained weights contain the keys
`sfe`, `dfe.0..7.{sbb,rdb,cca}`, `dff_conv1`, `dff_conv3`, `dff_pa`, `iir_conv` and none of
`attn`, `qkv`, `relative_position`, `window`, `num_heads` or `softmax` — 138 parameter tensors,
matching `models/lisn.py` exactly.
