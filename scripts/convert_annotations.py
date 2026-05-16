"""
Convert ExDark annotation format to YOLO format.
Run ONCE before anything else.

ExDark annotation line format:
% ExDark  <Class>  <left>  <top>  <width>  <height>  <extra...>

YOLO format needed:
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Usage: python scripts/convert_annotations.py
"""
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CLASS_MAP = {
    'Bicycle': 0, 'Boat': 1, 'Bottle': 2, 'Bus': 3,
    'Car': 4, 'Cat': 5, 'Chair': 6, 'Cup': 7,
    'Dog': 8, 'Motorbike': 9, 'People': 10, 'Table': 11
}

# Adjust these paths to match your ExDark download structure
ANNO_DIR   = 'data/ExDark/ExDark_Annno'   # folder with .txt annotation files
IMAGE_DIR  = 'data/ExDark/ExDark'          # folder with actual images
OUTPUT_DIR = 'data/ExDark/labels/all'     # YOLO labels written here

def get_base_name(stem: str) -> str:
    """
    Remove image extension from annotation stem.
    Example: "2015_00001.png" -> "2015_00001"
    """
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        if stem.lower().endswith(ext):
            return stem[:-len(ext)]
    return stem

def convert(anno_dir: str, image_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    anno_files = list(Path(anno_dir).rglob('*.txt'))
    skipped = 0
    converted = 0

    for anno_path in tqdm(anno_files, desc='Converting annotations'):
        # Get base name without any extension
        raw_stem = anno_path.stem          # e.g., "2015_00001.png"
        base_name = get_base_name(raw_stem)  # e.g., "2015_00001"

        # Find matching image (any extension, recursively)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = Path(image_dir) / (base_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            # Search recursively in subfolders (ExDark has class subdirs)
            matches = list(Path(image_dir).rglob(base_name + '.*'))
            if matches:
                img_path = matches[0]

        if img_path is None:
            skipped += 1
            continue

        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        yolo_lines = []
        with open(anno_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue

                # Handle both "% ExDark Class left top w h" and "Class left top w h"
                if parts[0] == '%' and len(parts) >= 7:
                    class_name = parts[2]
                    try:
                        left, top, w, h = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    except (ValueError, IndexError):
                        continue
                else:
                    class_name = parts[0]
                    try:
                        left, top, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except (ValueError, IndexError):
                        continue

                if class_name not in CLASS_MAP:
                    continue

                x_center = max(0.0, min(1.0, (left + w / 2) / img_w))
                y_center = max(0.0, min(1.0, (top  + h / 2) / img_h))
                w_norm   = max(0.0, min(1.0, w / img_w))
                h_norm   = max(0.0, min(1.0, h / img_h))

                yolo_lines.append(
                    f"{CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

        if yolo_lines:
            out = Path(output_dir) / (base_name + '.txt')
            out.write_text('\n'.join(yolo_lines))
            converted += 1

    print(f"Done. Converted {converted} files, skipped {skipped} files (image not found).")

if __name__ == '__main__':
    convert(ANNO_DIR, IMAGE_DIR, OUTPUT_DIR)