## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Set up the project environment, configure the repository structure, and prepare the ExDark dataset for model training and evaluation. This is an AFK slice. You will install all dependencies and run the provided scripts to convert the ExDark annotations to YOLO format and split the dataset into train/val/test directories.

## Acceptance criteria

- [ ] Project root contains `requirements.txt` and `exdark.yaml` with the contents provided in the PRD.
- [ ] Virtual environment is created and dependencies are installed via `pip install -r requirements.txt`.
- [ ] ExDark dataset images and annotations are downloaded and placed into `data/ExDark/`.
- [ ] `scripts/convert_annotations.py` is implemented and executed, generating YOLO labels.
- [ ] `scripts/split_dataset.py` is implemented and executed, successfully creating the 80/10/10 train/val/test splits in `data/ExDark/images/` and `data/ExDark/labels/`.

## Blocked by

None - can start immediately

## User stories addressed

- Person A Day 1 Tasks: Environment setup and Dataset splitting.
