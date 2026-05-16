"""
Low-light image enhancement — Framework #2: OpenCV.

Two techniques:
  CLAHE  — Contrast Limited Adaptive Histogram Equalization (LAB color space)
  Gamma  — Power-law pixel transform to raise luminance

Both accept and return BGR numpy arrays (OpenCV native format).
"""
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    Apply CLAHE to BGR image via LAB color space.
    Enhances luminance channel only — preserves color balance.

    clip_limit: 2.0-4.0 recommended. Higher = more aggressive contrast.
    tile_size:  8 is standard. Smaller = finer local adaptation.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced_lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """
    Power-law gamma correction. gamma > 1.0 brightens (use 1.5-2.5 for low-light).
    Uses a precomputed LUT for speed — O(1) per pixel.
    """
    if image is None or image.size == 0:
        return image
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image, lut)


def enhance_image(image: np.ndarray, method: str = 'both', **kwargs) -> np.ndarray:
    """
    Unified enhancement interface for dashboard and evaluation.

    method: 'clahe' | 'gamma' | 'both'
    kwargs: clip_limit, tile_size, gamma (passed to sub-functions)
    """
    method = method.lower()
    if method == 'clahe':
        return apply_clahe(
            image,
            clip_limit=kwargs.get('clip_limit', 3.0),
            tile_size=kwargs.get('tile_size', 8)
        )
    elif method == 'gamma':
        return apply_gamma_correction(image, gamma=kwargs.get('gamma', 1.8))
    elif method == 'both':
        img = apply_gamma_correction(image, gamma=kwargs.get('gamma', 1.8))
        return apply_clahe(img, clip_limit=kwargs.get('clip_limit', 3.0))
    else:
        raise ValueError(f"Unknown enhancement method: '{method}'. Use 'clahe', 'gamma', or 'both'.")


def compute_brightness(image: np.ndarray) -> float:
    """Return mean pixel luminance in [0, 255]. Used in dashboard brightness display."""
    if image is None or image.size == 0:
        return 0.0
    return float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))


def compute_contrast(image: np.ndarray) -> float:
    """Return pixel std deviation — proxy for contrast level."""
    if image is None or image.size == 0:
        return 0.0
    return float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
