"""
Full evaluation pipeline. Run this ONCE after dataset is set up and
fine-tuned weights are placed in models/best.pt.
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import time

# ============= CONFIGURATION =============
DATA_YAML = 'exdark.yaml'
MODEL_NANO = 'yolov8n.pt'
MODEL_BEST = 'models/best.pt'
N_FPS_SAMPLE = 20  # Reduced for faster testing
N_ENH_SAMPLE = 30  # Reduced for faster testing

# Find test directory automatically
def find_test_dir():
    possible_paths = [
        Path('data/ExDark/images/test'),
        Path('../data/ExDark/images/test'),
        Path('D:/CVLabFinal/Adaptive-AI-Surveillance/data/ExDark/images/test'),
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

TEST_DIR = find_test_dir()

# ============= HELPER FUNCTIONS =============

def compute_brightness(img):
    """Compute average brightness of an image"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(img))

def enhance_image(img, method='both'):
    """Enhance image using CLAHE and Gamma correction"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Apply Gamma correction
    gamma = 1.5
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    l_enhanced = cv2.LUT(l_enhanced, look_up_table)
    
    # Merge back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_objects(model, img):
    """Run detection and extract confidences"""
    results = model(img)
    confidences = []
    if len(results) > 0 and results[0].boxes is not None:
        confidences = results[0].boxes.conf.cpu().numpy().tolist()
    return confidences

def check_paths():
    """Verify all required paths exist"""
    print("\n Checking paths...")
    issues = []
    
    if TEST_DIR is None:
        issues.append(" Test directory not found")
        print("    Please update TEST_DIR path in the script")
    else:
        print(f"    Test images: {TEST_DIR}")
    
    if not Path(MODEL_BEST).exists():
        issues.append(f" Fine-tuned model not found: {MODEL_BEST}")
        print("    Make sure models/best.pt exists")
    else:
        print(f"   Fine-tuned model: {MODEL_BEST}")
    
    if not Path(MODEL_NANO).exists():
        print(f"    Downloading YOLOv8n.pt...")
        YOLO(MODEL_NANO)
    
    if not Path(DATA_YAML).exists():
        issues.append(f" Data YAML not found: {DATA_YAML}")
    else:
        print(f"    Data YAML: {DATA_YAML}")
    
    return len(issues) == 0

def measure_fps(model_path: str, n: int = N_FPS_SAMPLE) -> float:
    """Measure FPS on test images"""
    if TEST_DIR is None:
        return 0.0
    
    model = YOLO(model_path)
    images = list(TEST_DIR.glob('*.jpg')) + list(TEST_DIR.glob('*.png'))
    images = images[:n]
    
    if not images:
        return 0.0
    
    times = []
    for img_path in tqdm(images, desc=f'FPS [{Path(model_path).name}]'):
        img = cv2.imread(str(img_path))
        if img is not None:
            start = time.time()
            model(img)
            times.append(time.time() - start)
    
    if times:
        fps = 1.0 / np.mean(times)
        return round(fps, 1)
    return 0.0

def save_results(results: dict, output_dir: Path = Path('results/metrics')):
    """Save evaluation results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'eval_results.json'
    
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = {k: convert(v) for k, v in results.items()}
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"    Results saved to {output_path}")
    return output_path

# ============= MAIN EVALUATION =============

if __name__ == '__main__':
    print("=" * 70)
    print("FULL EVALUATION — ExDark Test Set")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not check_paths():
        print("\n Some paths are missing. Please fix and run again.")
        exit(1)
    
    results = {}
    
    # Step 1: Evaluate YOLOv8n pretrained
    print("\n[1/4]  Evaluating YOLOv8n...")
    model_nano = YOLO(MODEL_NANO)
    metrics_nano = model_nano.val(data=DATA_YAML, split='test')
    results['yolov8n'] = {
        'mAP50': float(metrics_nano.box.map50),
        'mAP50-95': float(metrics_nano.box.map),
        'precision': float(metrics_nano.box.mp),
        'recall': float(metrics_nano.box.mr)
    }
    print(f"      mAP50={results['yolov8n']['mAP50']:.3f}")
    
    # Step 2: Evaluate fine-tuned model
    print("\n[2/4]  Evaluating fine-tuned model...")
    model_best = YOLO(MODEL_BEST)
    metrics_best = model_best.val(data=DATA_YAML, split='test')
    results['best_ft'] = {
        'mAP50': float(metrics_best.box.map50),
        'mAP50-95': float(metrics_best.box.map),
        'precision': float(metrics_best.box.mp),
        'recall': float(metrics_best.box.mr)
    }
    print(f"      mAP50={results['best_ft']['mAP50']:.3f}")
    
    # Step 3: Enhancement impact evaluation
    print("\n[3/4]  Evaluating enhancement impact...")
    enh_images = list(TEST_DIR.glob('*.jpg')) + list(TEST_DIR.glob('*.png'))
    enh_images = enh_images[:N_ENH_SAMPLE]
    
    conf_raw = []
    conf_enh = []
    bright_raw = []
    bright_enh = []
    
    for img_path in tqdm(enh_images, desc='Enhancement Evaluation'):
        img = cv2.imread(str(img_path))
        if img is None: 
            continue
        
        # Original image
        b_raw = compute_brightness(img)
        c_raw_list = detect_objects(model_best, img)
        c_raw = np.mean(c_raw_list) if c_raw_list else 0.0
        
        # Enhanced image
        img_enh = enhance_image(img)
        b_enh = compute_brightness(img_enh)
        c_enh_list = detect_objects(model_best, img_enh)
        c_enh = np.mean(c_enh_list) if c_enh_list else 0.0
        
        bright_raw.append(b_raw)
        bright_enh.append(b_enh)
        conf_raw.append(c_raw)
        conf_enh.append(c_enh)
    
    avg_conf_raw = float(np.mean(conf_raw)) if conf_raw else 0.0
    avg_conf_enh = float(np.mean(conf_enh)) if conf_enh else 0.0
    
    results['enhancement'] = {
        'avg_conf_raw': round(avg_conf_raw, 4),
        'avg_conf_enhanced': round(avg_conf_enh, 4),
        'conf_improvement_pct': round(((avg_conf_enh - avg_conf_raw) / max(avg_conf_raw, 1e-6)) * 100, 2),
        'avg_brightness_raw': round(float(np.mean(bright_raw)) if bright_raw else 0.0, 2),
        'avg_brightness_enh': round(float(np.mean(bright_enh)) if bright_enh else 0.0, 2)
    }
    print(f"      Conf before: {results['enhancement']['avg_conf_raw']:.3f} → after: {results['enhancement']['avg_conf_enhanced']:.3f}")
    
    # Step 4: FPS benchmark
    print("\n[4/4]  FPS benchmark...")
    results['fps_nano'] = measure_fps(MODEL_NANO)
    results['fps_best'] = measure_fps(MODEL_BEST)
    print(f"      YOLOv8n: {results['fps_nano']:.1f} FPS")
    print(f"      Fine-tuned: {results['fps_best']:.1f} FPS")
    
    # Calculate improvements
    def calc_imp(after, before):
        return ((after - before) / max(before, 1e-6)) * 100
    
    results['impact'] = {
        'mAP50_before': results['yolov8n']['mAP50'],
        'mAP50_after': results['best_ft']['mAP50'],
        'mAP50_improvement_pct': round(calc_imp(results['best_ft']['mAP50'], results['yolov8n']['mAP50']), 2),
        'precision_before': results['yolov8n']['precision'],
        'precision_after': results['best_ft']['precision'],
        'precision_improvement_pct': round(calc_imp(results['best_ft']['precision'], results['yolov8n']['precision']), 2),
        'recall_before': results['yolov8n']['recall'],
        'recall_after': results['best_ft']['recall'],
        'recall_improvement_pct': round(calc_imp(results['best_ft']['recall'], results['yolov8n']['recall']), 2)
    }
    
    # Save results
    save_results(results)
    
    # Final summary
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n SUMMARY:")
    print(f"   mAP50:      {results['yolov8n']['mAP50']:.3f} → {results['best_ft']['mAP50']:.3f} ({results['impact']['mAP50_improvement_pct']:+.1f}%)")
    print(f"   Precision:  {results['yolov8n']['precision']:.3f} → {results['best_ft']['precision']:.3f}")
    print(f"   Recall:     {results['yolov8n']['recall']:.3f} → {results['best_ft']['recall']:.3f}")
    print(f"   FPS:        {results['fps_nano']:.1f} → {results['fps_best']:.1f}")
    print(f"\n Results saved to: results/metrics/eval_results.json")
    print("=" * 70)