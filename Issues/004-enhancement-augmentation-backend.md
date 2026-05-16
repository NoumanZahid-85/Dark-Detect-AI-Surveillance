## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Implement the low-light image enhancement pipeline (OpenCV CLAHE + Gamma) and the test-time augmentation (TTA) pipeline using Albumentations to simulate and handle low-light conditions.

## Acceptance criteria

- [ ] `utils/enhancement.py` is implemented with `apply_clahe`, `apply_gamma_correction`, and `enhance_image` functions.
- [ ] `utils/augmentation.py` is implemented to provide the low-light simulation pipeline and test-time augmentation logic.
- [ ] Test the enhancement backend locally by simulating a low-light image and then running the enhancement on it to observe visual improvements.

## Blocked by

- Blocked by `Issues/002-core-detection-backend.md`

## User stories addressed

- Person A Day 2 Tasks: Enhancement and Augmentation Implementation.
