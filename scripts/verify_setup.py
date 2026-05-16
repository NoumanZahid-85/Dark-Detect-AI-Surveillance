"""
Pre-demo sanity check. Run this before the lab presentation.
Prints pass/fail for every critical requirement.

Usage: python scripts/verify_setup.py
"""
from pathlib import Path
import sys

checks = []

def check(name, condition, fix=''):
    status = 'PASS' if condition else 'FAIL'
    checks.append((status, name, fix))

# Frameworks importable
try:
    # pyrefly: ignore [missing-import]
    import ultralytics; check('Framework 1: ultralytics importable', True)
except ImportError:
    check('Framework 1: ultralytics importable', False, 'pip install ultralytics')

try:
    import cv2; check('Framework 2: OpenCV importable', True)
except ImportError:
    check('Framework 2: OpenCV importable', False, 'pip install opencv-python')

try:
    # pyrefly: ignore [missing-import]
    import albumentations; check('Framework 3: Albumentations importable', True)
except ImportError:
    check('Framework 3: Albumentations importable', False, 'pip install albumentations')

# Dataset
check('ExDark test images exist',  Path('data/ExDark/images/test').exists() or Path('../data/ExDark/images/test').exists() or Path('D:/CVLabFinal/Adaptive-AI-Surveillance/data/ExDark/images/test').exists(),
      'Ensure dataset is extracted and split.')
check('exdark.yaml exists',        Path('exdark.yaml').exists(),
      'Create exdark.yaml')

# Model weights
check('Fine-tuned best.pt exists', Path('models/best.pt').exists(),
      'Download from Kaggle and place in models/')

# Evaluation output
check('eval_results.json exists',  Path('results/metrics/eval_results.json').exists(),
      'Run python scripts/run_eval.py')
check('Charts generated',          len(list(Path('results/charts').glob('*.png'))) >= 1,
      'Run python scripts/run_eval.py or python generate_chart.py')

# App
try:
    import streamlit; check('Streamlit importable', True)
except ImportError:
    check('Streamlit importable', False, 'pip install streamlit')

print("\n" + "=" * 55)
print("PRE-DEMO VERIFICATION")
print("=" * 55)
for status, name, fix in checks:
    icon = '✓' if status == 'PASS' else '✗'
    print(f"  {icon} [{status}]  {name}")
    if status == 'FAIL' and fix:
        print(f"           Fix: {fix}")

fails = sum(1 for s, _, _ in checks if s == 'FAIL')
print("=" * 55)
print(f"  {len(checks) - fails}/{len(checks)} checks passed")
if fails == 0:
    print("  Ready for demo.")
else:
    print(f"  {fails} issue(s) must be fixed before presenting.")
print("=" * 55)
sys.exit(fails)
