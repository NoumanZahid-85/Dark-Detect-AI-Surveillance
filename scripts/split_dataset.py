"""
Split converted YOLO labels + images into train/val/test.
Run ONCE after convert_annotations.py.

Usage: python scripts/split_dataset.py
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

LABELS_ALL = Path('data/ExDark/labels/all')
IMAGE_SRC  = Path('data/ExDark/ExDark')      # Adjust to your ExDark image folder
BASE       = Path('data/ExDark')

SPLITS = {'train': 0.80, 'val': 0.10, 'test': 0.10}
SEED   = 42

random.seed(SEED)

label_files = list(LABELS_ALL.glob('*.txt'))
random.shuffle(label_files)

n = len(label_files)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)
splits  = {
    'train': label_files[:n_train],
    'val':   label_files[n_train:n_train + n_val],
    'test':  label_files[n_train + n_val:],
}

for split, files in splits.items():
    (BASE / 'images' / split).mkdir(parents=True, exist_ok=True)
    (BASE / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for lf in tqdm(files, desc=f'Copying {split}'):
        stem = lf.stem          # e.g., "2015_00001"
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            matches = list(IMAGE_SRC.rglob(stem + ext))
            if matches:
                img_path = matches[0]
                break

        if img_path is None:
            continue

        shutil.copy2(img_path, BASE / 'images' / split / img_path.name)
        shutil.copy2(lf,       BASE / 'labels' / split / lf.name)

print(f"Split complete: {n_train} train / {n_val} val / {n - n_train - n_val} test")