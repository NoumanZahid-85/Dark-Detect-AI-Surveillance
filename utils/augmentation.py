"""
Low-light simulation and test-time augmentation — Framework #3: Albumentations.

Purposes:
  1. Simulate low-light on normal images (for testing enhancement pipeline)
  2. Provide test-time augmentation (TTA) for robustness eval
  3. Explicitly satisfies the "3rd CV framework" requirement

NOTE: Albumentations expects RGB input, returns RGB. Convert from/to BGR at boundaries.
"""
import cv2
import numpy as np
import albumentations as A


# ── Low-light simulation ───────────────────────────────────────────────────────

_SEVERITY_PARAMS = {
    'mild':   {'gamma_limit': (65, 80),  'noise_var_limit': (5, 15)},
    'medium': {'gamma_limit': (35, 60),  'noise_var_limit': (15, 30)},
    'severe': {'gamma_limit': (10, 35),  'noise_var_limit': (30, 50)},
}


def get_low_light_pipeline(severity: str = 'medium') -> A.Compose:
    """
    Returns an Albumentations pipeline that simulates low-light conditions.
    RandomGamma + GaussNoise + slight brightness/contrast reduction.
    """
    p = _SEVERITY_PARAMS.get(severity, _SEVERITY_PARAMS['medium'])
    return A.Compose([
        A.RandomGamma(gamma_limit=p['gamma_limit'], p=1.0),
        A.GaussNoise(var_limit=p['noise_var_limit'], p=0.75),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.35, -0.10),
            contrast_limit=(-0.20, 0.0),
            p=0.80
        ),
    ])


def simulate_low_light(image_bgr: np.ndarray, severity: str = 'medium') -> np.ndarray:
    """
    Darken a BGR image to simulate low-light surveillance conditions.
    Uses Albumentations (Framework #3) internally.

    Args:
        image_bgr: OpenCV BGR numpy array
        severity: 'mild' | 'medium' | 'severe'
    Returns:
        Darkened BGR numpy array
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pipeline = get_low_light_pipeline(severity)
    result = pipeline(image=rgb)['image']
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ── Test-time augmentation ─────────────────────────────────────────────────────

def get_tta_transforms() -> list:
    """
    Returns list of Albumentations transforms for test-time augmentation.
    Run inference on each transformed version, then average confidences.
    """
    return [
        A.Compose([]),                                                     # Original
        A.Compose([A.HorizontalFlip(p=1.0)]),                             # Mirrored
        A.Compose([A.RandomBrightnessContrast(0.10, 0.10, p=1.0)]),       # Brighter
        A.Compose([A.RandomGamma(gamma_limit=(80, 100), p=1.0)]),          # Slight darken
    ]


def apply_tta_transform(image_bgr: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply a single TTA transform to a BGR image. Returns BGR."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = transform(image=rgb)['image']
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ── Dataset-level simulation (used in run_eval.py) ────────────────────────────

def batch_simulate_low_light(
    image_paths: list,
    output_dir: str,
    severity: str = 'medium'
) -> None:
    """
    Apply low-light simulation to a list of images, save to output_dir.
    Used to create a simulated-dark test set from COCO images if needed.
    """
    import os
    from pathlib import Path
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    pipeline = get_low_light_pipeline(severity)

    for img_path in tqdm(image_paths, desc=f'Simulating ({severity})'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        darkened = cv2.cvtColor(pipeline(image=rgb)['image'], cv2.COLOR_RGB2BGR)
        out_path = Path(output_dir) / Path(img_path).name
        cv2.imwrite(str(out_path), darkened)
