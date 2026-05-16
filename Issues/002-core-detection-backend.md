## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Implement the core YOLOv8 inference wrapper. This backend module will handle image inference, bounding box extraction, confidence scores, and FPS calculation. It serves as the foundation for the frontend and evaluation scripts. 

## Acceptance criteria

- [ ] `utils/__init__.py` is created to make the utils directory a package.
- [ ] `utils/detection.py` is implemented exactly as provided in the PRD, featuring the `YOLODetector` class.
- [ ] `YOLODetector` correctly auto-detects device (CPU/CUDA) without hardcoding `device=0`.
- [ ] A temporary script successfully instantiates `YOLODetector` and runs inference on 5 test images, confirming bounding boxes are drawn, FPS is printed, and object counts are returned.

## Blocked by

- Blocked by `Issues/001-environment-dataset-prep.md`

## User stories addressed

- Person A Day 1 Tasks: Core Detection Implementation.
