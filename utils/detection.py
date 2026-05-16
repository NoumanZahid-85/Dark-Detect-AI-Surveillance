"""
YOLOv8 detection wrapper — Framework #1: Ultralytics.

Handles:
  - Single image inference with timing
  - Video frame processing
  - Rolling FPS average (30-frame window)
  - Dataset evaluation → mAP50, precision, recall
  - Auto device detection (GPU if available, CPU fallback)
"""
import time
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

# BGR colors per class — consistent across all visualizations
CLASS_COLORS = [
    (255, 80,  80),  # Bicycle  — blue
    (80,  255, 80),  # Boat     — green
    (80,  80,  255), # Bottle   — red
    (255, 255, 80),  # Bus      — cyan
    (255, 80,  255), # Car      — magenta
    (80,  255, 255), # Cat      — yellow
    (200, 140, 80),  # Chair    — teal
    (140, 200, 80),  # Cup      — lime
    (80,  140, 200), # Dog      — orange
    (200, 80,  140), # Motorbike
    (140, 80,  200), # People
    (80,  200, 140), # Table
]


def _auto_device() -> str:
    """Return 'cuda' if NVIDIA GPU available, else 'cpu'. Never hardcode."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class YOLODetector:
    """
    Clean wrapper around ultralytics YOLO.
    Use this everywhere — never call YOLO() directly in app.py or eval scripts.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = _auto_device()
        self.model = YOLO(model_path)
        self._fps_window = deque(maxlen=30)   # Rolling 30-frame FPS average
        self._warmup_done = False

    def _warmup(self, shape=(640, 640)):
        """Run one dummy inference to initialize CUDA and model graph."""
        dummy = np.zeros((*shape, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
        self._warmup_done = True

    def infer_image(self, image: np.ndarray) -> dict:
        """
        Run inference on a single BGR image.

        Returns:
          boxes:        [[x1,y1,x2,y2], ...]
          classes:      ['Car', 'People', ...]
          confidences:  [0.87, 0.64, ...]
          fps:          float (rolling 30-frame average)
          count:        int
          annotated:    BGR image with drawn boxes
          class_counts: {'Car': 2, 'People': 3, ...}
        """
        if not self._warmup_done:
            self._warmup()

        t0 = time.perf_counter()
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        elapsed = time.perf_counter() - t0

        self._fps_window.append(1.0 / elapsed if elapsed > 0 else 0.0)

        boxes, classes, confidences = [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = EXDARK_CLASSES[cls_id] if cls_id < len(EXDARK_CLASSES) \
                     else self.model.names.get(cls_id, str(cls_id))
            boxes.append([x1, y1, x2, y2])
            classes.append(name)
            confidences.append(conf)

        class_counts = {}
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            'boxes':       boxes,
            'classes':     classes,
            'confidences': confidences,
            'fps':         self.get_avg_fps(),
            'count':       len(boxes),
            'annotated':   self._draw(image.copy(), boxes, classes, confidences),
            'class_counts': class_counts,
        }

    def _draw(self, image, boxes, classes, confidences) -> np.ndarray:
        """Draw bounding boxes with class labels on image."""
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS[EXDARK_CLASSES.index(cls)] \
                    if cls in EXDARK_CLASSES else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - lh - 6), (x1 + lw + 2, y1), color, -1)
            cv2.putText(image, label, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def get_avg_fps(self) -> float:
        """Rolling 30-frame FPS average. Returns 0.0 if no frames yet."""
        return sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0.0

    def evaluate_on_dataset(self, data_yaml: str, split: str = 'test') -> dict:
        """
        Run ultralytics built-in validation on a dataset split.
        This is the ONLY valid way to get mAP50/precision/recall for the report.

        Returns:
          mAP50, mAP50_95, precision, recall, model
        """
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            device=self.device,
            verbose=False
        )
        return {
            'mAP50':     round(float(metrics.box.map50), 4),
            'mAP50_95':  round(float(metrics.box.map),   4),
            'precision': round(float(metrics.box.mp),    4),
            'recall':    round(float(metrics.box.mr),    4),
            'model':     self.model_path,
        }
