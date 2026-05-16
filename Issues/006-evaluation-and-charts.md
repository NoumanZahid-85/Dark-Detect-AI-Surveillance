## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Fine-tune the YOLOv8 model on the ExDark dataset (via Kaggle) and execute the evaluation script to calculate mAP50, precision, recall, and overall improvement percentage.

## Acceptance criteria

- [ ] Download the `best.pt` fine-tuned weights from Kaggle and place them in the `models/` directory.
- [ ] Create `scripts/run_eval.py` to evaluate the model on the ExDark test set.
- [ ] Verify that `eval_results.json` is generated with metrics: mAP50, precision, recall, and improvement percentage.
- [ ] Implement `utils/visualization.py` and generate all 4 required evaluation charts in `results/charts/`.

## Blocked by

- Blocked by `Issues/004-enhancement-augmentation-backend.md`

## User stories addressed

- Person A Day 2 Tasks: Evaluation and visualization.
- Person A Day 3 Tasks: Best weights placement.
