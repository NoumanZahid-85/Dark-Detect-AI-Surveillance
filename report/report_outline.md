# Comprehensive Report Outline: Adaptive AI Surveillance

> **Instruction for AI Report Generator:** This document provides the exact structure, key talking points, specific data locations, and expected artifacts needed to generate a professional, academic-grade 4-page report for the "Adaptive AI Surveillance" project (COMSATS CV Lab Final SP2026). Ensure professional, analytical, and academic tone throughout.

---

## Page 1: Problem Statement & Application Domain

### 1.1 Introduction and The "Domain Gap" Problem
*   **Context:** Introduce object detection in surveillance systems.
*   **The Problem:** Standard object detectors (e.g., YOLOv8n) are typically trained on the MS COCO dataset, which consists of high-quality, well-lit images (less than 2% low-light). When these models are deployed in night-time or low-light scenarios, they suffer from a severe "domain shift" or "domain gap."
*   **Symptoms:** Feature distribution mismatch (dark pixels vs. lit pixels), resulting in low confidence scores, high false negatives, and poor overall Mean Average Precision (mAP).

### 1.2 Application Domain
*   **Target Scenarios:** Parking lot surveillance, indoor security, night-time traffic monitoring, and low-light urban environments.
*   **Significance:** Critical for 24/7 autonomous monitoring systems where relying strictly on infrared/thermal cameras is cost-prohibitive.

### 1.3 Project Options Addressed (Rubric Criteria)
*   *Explicitly state that this project fulfills the following assignment options:*
*   **Option 2 (Real-Time Detection):** Achieving >15 FPS on CPU and ~30 FPS on GPU using optimized YOLOv8 architectures.
*   **Option 4 (Domain Adaptation):** Utilizing transfer learning to fine-tune on the ExDark dataset, combined with CLAHE and Gamma Correction for dynamic luminance recovery.
*   **Option 8 (Dataset Benchmarking):** Providing extensive, data-driven comparisons (mAP, Precision, Recall) between baseline YOLOv8n and the fine-tuned domain-specific model.

---

## Page 2: Methodology & System Architecture

### 2.1 The ExDark Dataset
*   **Overview:** Describe the Exclusively Dark (ExDark) dataset (7,363 images, 12 specific classes).
*   **Why ExDark?:** It is the academic standard for low-light benchmarking, solving the class-imbalance and illumination deficiencies of COCO. Mention the 80/10/10 Train/Val/Test split.

### 2.2 System Architecture & The 3-Framework Pipeline
*   *Detail the flow of data through the system:* Raw Image -> Enhancement (OpenCV) -> Augmentation Simulation (Albumentations) -> Inference (Ultralytics) -> Telemetry/Analytics.
*   **Framework 1: Ultralytics (YOLOv8):** Chosen for its state-of-the-art speed-to-accuracy ratio compared to Detectron2 or Faster R-CNN. Essential for real-time edge deployment.
*   **Framework 2: OpenCV (Enhancement):** 
    *   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied via LAB color space to enhance the luminance channel (L) without amplifying noise or distorting color balance (A, B).
    *   **Gamma Correction:** Power-law transform to boost dark pixel intensity.
*   **Framework 3: Albumentations (Simulation):** Used to inject synthetic noise (`GaussNoise`) and adjust gamma (`RandomGamma`) to simulate varying severities of darkness (mild, medium, severe) for robustness testing.

---

## Page 3: Experimental Results & Benchmarking

> **Instruction for AI:** Pull the exact numerical data for the tables below from `results/metrics/eval_results.json`. Do not invent numbers. 

### 3.1 Model Comparison on ExDark Test Set
*   Provide a narrative explaining that fine-tuning (transfer learning) allowed the model to retain geometric understanding while adapting its thresholds for dark pixels.
*   **Include Table 1: Model Benchmark Comparison**
    *   Columns: Metric | Standard YOLOv8n | Fine-Tuned Model
    *   Rows: mAP50, mAP50-95, Precision, Recall, Inference Speed (FPS).
*   **Chart Embed:** Reference or embed `results/charts/model_comparison.png`.

### 3.2 Enhancement Impact Analysis
*   Discuss how preprocessing the raw images before passing them to the baseline model affects detection confidence.
*   **Include Table 2: CLAHE + Gamma Enhancement Impact**
    *   Columns: Metric | Raw Image | Enhanced Image | Improvement (%)
    *   Rows: Average Confidence, Average Brightness Shift.
*   **Chart Embed:** Reference or embed `results/charts/confidence_distribution.png` and `results/charts/class_distribution.png`.

### 3.3 Computational Efficiency (FPS)
*   Briefly discuss edge-device feasibility. Mention the FPS benchmarks achieved during evaluation (e.g., 10-15 FPS on standard CPUs).
*   **Chart Embed:** Reference `results/charts/fps_comparison.png`.

---

## Page 4: Challenges, Conclusions, and Contributions

### 4.1 Engineering Challenges
*   **Dataset Format Conversion:** Developing robust scripts (`convert_annotations.py`) to parse custom ExDark text annotations and normalize bounding boxes into YOLO format.
*   **Hardware Constraints:** Managing GPU memory limits during fine-tuning on Kaggle, and optimizing the Streamlit dashboard for real-time video processing on CPU using frame-skipping (processing every 2nd frame).
*   **Enhancement vs. Noise:** Balancing Gamma Correction and CLAHE limits to avoid blowing out highlights or aggressively amplifying ISO noise in pitch-black images.

### 4.2 Conclusion
*   Summarize the massive percentage improvement in mAP50 achieved through fine-tuning.
*   Conclude that a hybrid approach—domain-specific fine-tuning coupled with dynamic OpenCV preprocessing—is the optimal strategy for low-light surveillance.

### 4.3 Team Contributions
*   **Person A (Backend, Modeling, Evaluation):** Engineered `utils/detection.py`, `utils/enhancement.py`, `utils/augmentation.py`. Handled ExDark dataset conversion, Kaggle YOLOv8 fine-tuning, and built the robust evaluation pipeline (`run_eval.py`).
*   **Person B (Frontend, Analytics, Documentation):** Developed the interactive Streamlit dashboard (`app.py`), crafted the visual analytics engine (`utils/visualization.py`), designed the system architecture UI, and authored the comprehensive `README.md` and Final Report.
