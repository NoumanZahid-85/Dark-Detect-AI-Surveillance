## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Add an enhancement toggle and a model selector to the Streamlit UI, allowing the user to seamlessly switch between raw/enhanced images and different YOLO model weights.

## Acceptance criteria

- [ ] Enhancement toggle is added to Tab 1.
- [ ] Toggling the enhancement control actively calls `enhance_image` from `utils.enhancement` and updates the displayed image.
- [ ] A model selector dropdown is added to allow choosing between `yolov8n.pt` and a fine-tuned `best.pt`.
- [ ] Inference correctly uses the selected model and visually updates the bounding boxes.

## Blocked by

- Blocked by `Issues/003-frontend-tab1-inference.md`
- Blocked by `Issues/004-enhancement-augmentation-backend.md`

## User stories addressed

- Person B Day 2 Tasks: Frontend toggles and selectors.
