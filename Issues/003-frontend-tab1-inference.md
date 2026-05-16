## Parent PRD

`FINAL_CURSOR_BLUEPRINT_V2.md`

## What to build

Create the Streamlit frontend skeleton (`app.py`) and wire up the first tab to accept image uploads, run them through the core detection backend, and display the annotated results.

## Acceptance criteria

- [ ] `app.py` is created with a basic Streamlit skeleton (sidebar, title, and 3 tabs).
- [ ] Image upload widget is added to Tab 1.
- [ ] Uploaded image is displayed on the screen.
- [ ] `YOLODetector` is imported from `utils.detection` and connected to the uploaded image.
- [ ] Tab 1 successfully runs inference and displays the annotated image alongside the calculated FPS.

## Blocked by

- Blocked by `Issues/002-core-detection-backend.md`

## User stories addressed

- Person B Day 1 Tasks: Streamlit skeleton and Image Upload wired to Detection.
