"""
Quality metrics for super-resolution evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr_ssim(sr_img, hr_img, border=4, data_range=255):
    """Compute PSNR and SSIM between SR and HR images.
    
    Args:
        sr_img: Super-resolved image (numpy uint8 or float32)
        hr_img: Ground truth HR image (numpy uint8 or float32)
        border: Pixels to crop from each edge before comparison
        data_range: 255 for uint8, 1.0 for float
    
    Returns:
        (psnr, ssim) tuple
    """
    if border > 0:
        sr_img = sr_img[border:-border, border:-border]
        hr_img = hr_img[border:-border, border:-border]

    psnr = peak_signal_noise_ratio(hr_img, sr_img, data_range=data_range)
    ssim = structural_similarity(hr_img, sr_img, data_range=data_range)
    return psnr, ssim
