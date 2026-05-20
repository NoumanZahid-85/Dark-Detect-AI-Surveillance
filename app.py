"""
Adaptive AI Surveillance Dashboard
COMSATS University Islamabad — Computer Vision Lab Final SP2026

Run: streamlit run app.py
"""
import json
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
import time

from utils.detection import YOLODetector
from utils.enhancement import enhance_image, compute_brightness, compute_contrast
from utils.augmentation import simulate_low_light
from utils.visualization import (
    plot_fps_comparison, plot_class_distribution,
    plot_confidence_distribution
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Low-Light AI Surveillance | COMSATS CV Lab",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ─── Global Dark Theme ─── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

.sidebar-header {
    background: linear-gradient(135deg, #022c22 0%, #0f2027 100%);
    padding: 20px 16px 16px;
    border-bottom: 1px solid #30363d;
    margin-bottom: 0;
}
.sidebar-logo {
    font-size: 28px;
    display: block;
    margin-bottom: 6px;
}
.sidebar-title {
    font-size: 15px;
    font-weight: 700;
    color: #10b981;
    letter-spacing: 0.3px;
    line-height: 1.3;
    margin: 0;
}
.sidebar-subtitle {
    font-size: 10px;
    color: #7d8590;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.sidebar-section {
    padding: 14px 16px 10px;
    border-bottom: 1px solid #21262d;
}
.sidebar-section-label {
    font-size: 9px;
    font-weight: 700;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #21262d;
}

/* ─── Status pills ─── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0d2818;
    border: 1px solid #238636;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    color: #3fb950;
    font-weight: 500;
    margin: 2px 0;
    width: 100%;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3fb950;
    box-shadow: 0 0 6px #3fb950;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ─── Metric cards ─── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #10b981 !important; }
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #10b981 !important;
}
[data-testid="stMetricLabel"] { color: #7d8590 !important; font-size: 12px !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ─── Tabs ─── */
[data-testid="stTabs"] { gap: 0; }
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    border-radius: 0;
    gap: 0;
    padding: 0 16px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #7d8590;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 20px;
    border-radius: 0;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #e6edf3; background: #21262d; }
.stTabs [aria-selected="true"] {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    background: transparent !important;
}

/* ─── Headers ─── */
h1 { font-size: 24px !important; font-weight: 700 !important; color: #e6edf3 !important; }
h2 { font-size: 18px !important; font-weight: 600 !important; color: #e6edf3 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; color: #c9d1d9 !important; }

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(16,185,129,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.4) !important;
}

/* ─── Selectbox / Inputs ─── */
[data-testid="stSelectbox"] > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
}
.stSlider [data-testid="stSlider"] > div { background: #21262d !important; }

/* ─── Dividers ─── */
hr { border-color: #21262d !important; }

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
}

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* ─── File uploader ─── */
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #10b981 !important; }

/* ─── Progress bar ─── */
.stProgress > div > div { background: #10b981 !important; border-radius: 4px; }

/* ─── Radio buttons ─── */
[data-testid="stRadio"] label { color: #c9d1d9 !important; font-size: 13px !important; }

/* ─── Toggle ─── */
[data-testid="stToggle"] span { font-size: 13px !important; }

/* ─── Alert/Info boxes ─── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ─── Caption text ─── */
[data-testid="stCaptionContainer"] { color: #7d8590 !important; font-size: 11px !important; }

/* ─── Page header ─── */
.page-header {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 50%, #0d1117 100%);
    border: 1px solid #065f46;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-header h1 {
    margin: 0 0 6px !important;
    font-size: 22px !important;
    background: linear-gradient(135deg, #34d399, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-header p {
    color: #7d8590;
    font-size: 13px;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #064e3b;
    color: #34d399;
    border: 1px solid #047857;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Low-Light AI Surveillance Dashboard</h1>
    <p>
        <span class="badge">YOLOv8</span>
        <span class="badge">OpenCV</span>
        <span class="badge">Albumentations</span>
        <span class="badge">ExDark Dataset</span>
        Real-time object detection &amp; enhancement analytics for low-light surveillance scenarios
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="sidebar-logo">🔦</span>
        <p class="sidebar-title">Low-Light AI Surveillance</p>
        <p class="sidebar-subtitle">COMSATS CV Lab &nbsp;·&nbsp; SP2026 Final Project</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Model ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Detection Model</div>', unsafe_allow_html=True)
    model_options = {
        "YOLOv8n  —  Fast / COCO pretrained": "yolov8n.pt",
        "YOLOv8s  —  Accurate / COCO pretrained": "yolov8s.pt",
    }
    if Path("models/best.pt").exists():
        model_options["Fine-tuned  —  ExDark (Best)"] = "models/best.pt"

    model_label = st.selectbox("Model", list(model_options.keys()), label_visibility="collapsed")
    model_file  = model_options[model_label]
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 2: Enhancement ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Enhancement  ·  OpenCV</div>', unsafe_allow_html=True)
    enhance_on     = st.toggle("Enable Low-Light Enhancement", value=False)
    enhance_method = st.selectbox(
        "Method",
        ["CLAHE", "Gamma Correction", "Both (CLAHE + Gamma)"],
        disabled=not enhance_on
    )
    method_map = {
        "CLAHE":                 "clahe",
        "Gamma Correction":      "gamma",
        "Both (CLAHE + Gamma)":  "both"
    }
    if enhance_on:
        st.caption("Enhancing luminance via LAB color space")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 3: Simulation ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Low-Light Simulation  ·  Albumentations</div>', unsafe_allow_html=True)
    simulate_on  = st.toggle("Simulate Low-Light Input", value=False)
    sim_severity = st.select_slider(
        "Severity",
        options=["mild", "medium", "severe"],
        value="medium",
        disabled=not simulate_on
    )
    if simulate_on:
        colors = {"mild": "#f0ad4e", "medium": "#e67e22", "severe": "#e74c3c"}
        st.caption(f"Darkening with {'RandomGamma + GaussNoise'} — {sim_severity} mode")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 4: Framework Status ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Active Frameworks</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-pill"><span class="status-dot"></span>Framework 1 — YOLOv8 (Ultralytics)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-pill"><span class="status-dot" style="background:{'#3fb950' if enhance_on else '#484f58'};box-shadow:{'0 0 6px #3fb950' if enhance_on else 'none'}"></span>
    Framework 2 — OpenCV{' · ACTIVE' if enhance_on else ''}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-pill"><span class="status-dot" style="background:{'#3fb950' if simulate_on else '#484f58'};box-shadow:{'0 0 6px #3fb950' if simulate_on else 'none'}"></span>
    Framework 3 — Albumentations{' · ACTIVE' if simulate_on else ''}</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 5: Quick Info ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Project Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#7d8590;line-height:1.6;">
    <b style="color:#c9d1d9;">Dataset</b><br>ExDark — 7,363 low-light images<br>
    12 classes: Car, People, Bicycle...<br><br>
    <b style="color:#c9d1d9;">Options Covered</b><br>
    Option 2 · Real-time detection<br>
    Option 4 · Domain adaptation<br>
    Option 8 · Benchmarking<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Model cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(model_path: str, conf: float) -> YOLODetector:
    return YOLODetector(model_path=model_path, conf_threshold=conf)

detector = load_detector(model_file, conf_thresh)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Detection", "Analytics & Results", "Model Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Real-Time Object Detection")
    st.markdown("Monitor environments in real-time across images, video files, or live webcam feeds.")
    
    input_mode = st.radio(
        "Input source", ["Upload Image", "Upload Video", "Webcam"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── Image mode ─────────────────────────────────────────────────────────────
    if input_mode == "Upload Image":
        uploaded = st.file_uploader("Upload a low-light image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            raw_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr   = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

            if simulate_on:
                img_bgr = simulate_low_light(img_bgr, severity=sim_severity)

            img_processed = img_bgr.copy()
            if enhance_on:
                img_processed = enhance_image(img_bgr, method=method_map[enhance_method])

            results = detector.infer_image(img_processed)

            # Layout
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input Stream" + (" (Enhanced)" if enhance_on else " (Original)"))
                display_img = img_processed if enhance_on else img_bgr
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                b_before = compute_brightness(img_bgr)
                b_after  = compute_brightness(img_processed)
                c_before = compute_contrast(img_bgr)
                c_after  = compute_contrast(img_processed)
                
                if enhance_on:
                    st.info(f"**Brightness:** {b_before:.0f} → {b_after:.0f} &nbsp;&nbsp;|&nbsp;&nbsp; **Contrast:** {c_before:.1f} → {c_after:.1f}")

            with col2:
                st.subheader("Detection Output")
                st.image(cv2.cvtColor(results['annotated'], cv2.COLOR_BGR2RGB), use_container_width=True)
                
            st.divider()
            
            # Metrics row
            st.subheader("Detection Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Objects Detected", results['count'])
            m2.metric("Processing FPS", f"{results['fps']:.1f}")
            avg_conf = np.mean(results['confidences']) if results['confidences'] else 0
            m3.metric("Average Confidence", f"{avg_conf:.3f}")
            m4.metric("Active Model", Path(model_file).stem)

            # Advanced visualizations
            if results['class_counts']:
                st.divider()
                st.subheader("Visual Analysis")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig_class = plot_class_distribution(results['class_counts'], save=False)
                    if fig_class:
                        st.pyplot(fig_class)
                
                with chart_col2:
                    if len(results['confidences']) > 0:
                        fig_conf = plot_confidence_distribution(results['confidences'], save=False)
                        if fig_conf:
                            st.pyplot(fig_conf)

    # ── Video mode ─────────────────────────────────────────────────────────────
    elif input_mode == "Upload Video":
        uploaded_vid = st.file_uploader("Upload a surveillance video (MP4/AVI)", type=["mp4", "avi", "mov"])
        if uploaded_vid:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            cap     = cv2.VideoCapture(tmp_path)
            total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Layout for video processing
            col_vid, col_metrics = st.columns([2, 1])
            
            with col_vid:
                stframe = st.empty()
                prog    = st.progress(0)
            
            with col_metrics:
                st.subheader("Live Telemetry")
                fps_ph  = st.empty()
                obj_ph  = st.empty()
                conf_ph = st.empty()
                
                st.markdown("---")
                chart_ph = st.empty() # Placeholder for live class distribution
                
            stop = st.button("Stop Processing")

            frame_idx    = 0
            all_classes  = defaultdict(int)
            all_confs    = []
            skip         = 2   # Process every 2nd frame for speed on CPU
            
            last_chart_update = time.time()

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % skip != 0:
                    continue

                if simulate_on:
                    frame = simulate_low_light(frame, severity=sim_severity)
                if enhance_on:
                    frame = enhance_image(frame, method=method_map[enhance_method])

                res = detector.infer_image(frame)
                
                # Update metrics
                for cls in res['classes']:
                    all_classes[cls] += 1
                all_confs.extend(res['confidences'])

                rgb = cv2.cvtColor(res['annotated'], cv2.COLOR_BGR2RGB)
                stframe.image(rgb, use_container_width=True)
                
                # Update live numbers
                fps_ph.metric("Live FPS", f"{res['fps']:.1f}")
                obj_ph.metric("Objects in Frame", res['count'])
                
                frame_conf = np.mean(res['confidences']) if res['confidences'] else 0
                conf_ph.metric("Frame Avg Confidence", f"{frame_conf:.3f}")
                
                # Update live chart every ~1 second to save CPU
                if time.time() - last_chart_update > 1.0 and all_classes:
                    fig = plot_class_distribution(dict(all_classes), save=False)
                    if fig:
                        chart_ph.pyplot(fig)
                    last_chart_update = time.time()
                
                if total > 0:
                    prog.progress(min(frame_idx / total, 1.0))

            cap.release()
            os.unlink(tmp_path)
            st.success(f"Processing complete: {frame_idx} frames analyzed.")
            
            if all_classes:
                fig = plot_class_distribution(dict(all_classes), save=False)
                if fig:
                    chart_ph.pyplot(fig)

    # ── Webcam mode ────────────────────────────────────────────────────────────
    elif input_mode == "Webcam":
        st.info("Click **Start Webcam** to begin live detection. Ensure your browser has camera permissions.")
        
        # Initialize session state for webcam
        if 'webcam_enabled' not in st.session_state:
            st.session_state.webcam_enabled = False
        if 'camera' not in st.session_state:
            st.session_state.camera = None
        
        col_button1, col_button2 = st.columns([1, 2])
        
        with col_button1:
            if st.button("▶ Start Webcam", use_container_width=True):
                st.session_state.webcam_enabled = True
        
        with col_button2:
            if st.button("⏹ Stop Webcam", use_container_width=True):
                st.session_state.webcam_enabled = False
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
        
        col_cam, col_metrics = st.columns([2, 1])
        
        with col_cam:
            stframe = st.empty()
            status_ph = st.empty()
            
        with col_metrics:
            st.subheader("Live Telemetry")
            fps_ph  = st.empty()
            obj_ph  = st.empty()
            conf_ph = st.empty()

        # Webcam stream processing
        if st.session_state.webcam_enabled:
            try:
                # Initialize camera if not already done
                if st.session_state.camera is None:
                    st.session_state.camera = cv2.VideoCapture(0)
                    st.session_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                    
                    # Verify camera opened successfully
                    if not st.session_state.camera.isOpened():
                        st.error("Webcam not found. Please check:")
                        st.markdown("""
                        - Camera is connected and not in use by another app
                        - Camera index is 0 (default). Try changing if multiple cameras exist
                        - System permissions allow camera access
                        - Try running: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`
                        """)
                        st.session_state.camera.release()
                        st.session_state.camera = None
                        st.session_state.webcam_enabled = False
                        st.stop()
                
                cap = st.session_state.camera
                
                # Capture single frame
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame from webcam. Device may have disconnected.")
                    st.session_state.camera.release()
                    st.session_state.camera = None
                    st.session_state.webcam_enabled = False
                    st.stop()
                
                # Process frame
                if enhance_on:
                    frame = enhance_image(frame, method=method_map[enhance_method])
                
                res = detector.infer_image(frame)
                
                # Display results
                stframe.image(
                    cv2.cvtColor(res['annotated'], cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
                
                status_ph.success("Webcam streaming active. Press 'Stop Webcam' to end.")
                
                # Update metrics
                fps_ph.metric("Live FPS", f"{res['fps']:.1f}")
                obj_ph.metric("Objects in Frame", res['count'])
                frame_conf = np.mean(res['confidences']) if res['confidences'] else 0
                conf_ph.metric("Frame Avg Confidence", f"{frame_conf:.3f}")
                
                # Auto-rerun for continuous streaming
                time.sleep(0.01)  # Small delay to control frame rate
                st.rerun()
                
            except Exception as e:
                st.error(f"Webcam error: {str(e)}")
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
                st.session_state.webcam_enabled = False
        else:
            # Clean up camera when disabled
            if st.session_state.camera is not None:
                try:
                    st.session_state.camera.release()
                except:
                    pass
                st.session_state.camera = None
            
            status_ph.info("⏸ Webcam stopped. Click Start Webcam to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics & Results
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Evaluation Results & Domain Analysis")
    st.markdown("Detailed breakdown of model performance on the ExDark dataset.")
    
    metrics_file = Path('results/metrics/eval_results.json')

    if not metrics_file.exists():
        st.warning("No evaluation results found. Run `python scripts/run_eval.py` to generate metrics.")
        st.code("python scripts/run_eval.py", language="bash")
    else:
        with open(metrics_file) as f:
            data = json.load(f)

        # ── Why Are Results So Wide? ───────────────────────────────────────────────
        st.markdown("### Domain Gap Explanation")
        st.info("""
        **Why is there such a massive difference between YOLOv8n (pretrained) and the Fine-Tuned Model?**
        
        The standard YOLOv8n model is trained on the COCO dataset, which contains high-quality, well-lit images. 
        When applied directly to the **ExDark** dataset (which consists entirely of low-light, noisy, and poor-contrast images), 
        the standard model suffers from severe **domain shift**.
        
        *   **Feature Distribution Mismatch:** The model has never learned to extract features from dark pixels, resulting in missed detections.
        *   **Class Imbalance:** COCO has 80 classes, while ExDark has 12 specific classes.
        
        **Fine-tuning** bridges this gap by adapting the model's weights specifically to low-light distributions, 
        resulting in the massive mAP (Mean Average Precision) jump shown below.
        """)
        
        st.divider()

        # ── Summary Metrics ────────────────────────────────────────────────────────
        st.subheader("Performance Summary (Test Set)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("mAP50 (Standard)", f"{data.get('yolov8n', {}).get('mAP50', 0):.3f}")
        col2.metric("mAP50 (Fine-Tuned)", f"{data.get('best_ft', {}).get('mAP50', 0):.3f}")
        
        impact = data.get('impact', {})
        gain = impact.get('mAP50_improvement_pct', 0)
        col3.metric("mAP Improvement", f"+{gain:,.0f}%" if gain > 0 else f"{gain:.1f}%")
        
        col4.metric("FPS (Standard CPU)", f"{data.get('fps_nano', 0):.1f}")
        col5.metric("FPS (Fine-Tuned)", f"{data.get('fps_best', 0):.1f}")

        # ── Enhancement Detail ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Image Enhancement Impact (CLAHE + Gamma Correction)")
        st.markdown("Measures how preprocessing algorithms improve detection confidence on raw dark images.")
        
        enh = data.get('enhancement', {})
        if enh:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Confidence (Raw)", f"{enh.get('avg_conf_raw', 0):.3f}")
            e2.metric("Confidence (Enhanced)", f"{enh.get('avg_conf_enhanced', 0):.3f}")
            e3.metric("Confidence Gain", f"{enh.get('conf_improvement_pct', 0):+.1f}%")
            e4.metric("Brightness Shift", f"{enh.get('avg_brightness_raw', 0):.0f} → {enh.get('avg_brightness_enh', 0):.0f}")
        else:
            st.warning("Enhancement impact metrics missing. Please re-run `python scripts/run_eval.py`.")

        # ── Charts ────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Detailed Visual Analytics")
        chart_dir = Path('results/charts')
        chart_files = sorted(chart_dir.glob('*.png'))
        
        if chart_files:
            # We want to display these large charts effectively.
            for chart in chart_files:
                st.image(str(chart), caption=chart.stem.replace('_', ' ').title(), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("Charts will appear here after running run_eval.py")

        with st.expander("Raw Evaluation Data (JSON)"):
            st.json(data)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Architecture Analysis")
    st.markdown("Comparing the foundational YOLO architecture with our fine-tuned domain-specific variant.")

    st.subheader("Technical Specifications")
    arch_df = pd.DataFrame({
        'Property':   ['Parameters', 'GFLOPs', 'Speed (CPU)', 'Training Data', 'Target Environment'],
        'Standard YOLOv8n':    ['3.2M', '8.7', '~25 FPS', 'COCO (80 classes)', 'Well-lit, general purpose'],
        'Fine-Tuned YOLOv8': ['3.2M', '8.7', '~25 FPS', 'ExDark (12 classes)', 'Low-light, night surveillance'],
    })
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.divider()
    
    st.subheader("Transfer Learning Approach")
    st.markdown("""
    Instead of training a model from scratch, this project utilizes **Transfer Learning**.
    1.  **Base Knowledge:** We begin with YOLOv8n, which has already learned fundamental concepts like edges, shapes, and textures from millions of COCO images.
    2.  **Domain Adaptation:** We continue training (fine-tuning) these pre-learned weights exclusively on the ExDark dataset.
    3.  **Result:** The model retains its ability to recognize complex shapes (cars, people) but adapts its threshold for pixel intensity, allowing it to effectively "see in the dark."
    """)

    if Path('results/metrics/eval_results.json').exists():
        with open('results/metrics/eval_results.json') as f:
            data = json.load(f)
        st.divider()
        st.subheader("Test Set Performance Comparison")
        
        nano = data.get('yolov8n', {})
        best = data.get('best_ft', {})
        
        cmp_df = pd.DataFrame({
            'Metric':    ['mAP50', 'mAP50-95', 'Precision', 'Recall'],
            'Standard YOLOv8n':   [nano.get('mAP50',0), nano.get('mAP50-95',0),
                          nano.get('precision',0), nano.get('recall',0)],
            'Fine-Tuned': [best.get('mAP50',0), best.get('mAP50-95',0),
                          best.get('precision',0), best.get('recall',0)],
        })
        
        # Style the dataframe to highlight the better values
        st.dataframe(
            cmp_df.style.highlight_max(subset=['Standard YOLOv8n', 'Fine-Tuned'], color='#1D9E75', axis=1),
            use_container_width=True,
            hide_index=True
        )
