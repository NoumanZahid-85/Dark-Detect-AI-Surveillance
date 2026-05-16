"""
Chart generation for dashboard and report.
All figures saved as PNG to results/charts/.
Use matplotlib Agg backend — required for Streamlit compatibility.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

CHART_DIR = Path('results/charts')
CHART_DIR.mkdir(parents=True, exist_ok=True)

_BLUE   = '#378ADD'
_GREEN  = '#1D9E75'
_RED    = '#E24B4A'
_AMBER  = '#EF9F27'
_PURPLE = '#7F77DD'


def _save(fig, name: str, save: bool) -> plt.Figure:
    fig.tight_layout()
    if save:
        fig.savefig(CHART_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    return fig


def plot_model_comparison(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Grouped bar chart: YOLOv8n vs fine-tuned model across metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = [c for c in df.columns if c in ['mAP50', 'mAP50-95', 'Precision', 'Recall']]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, df.iloc[0][metrics], w, label=df.index[0], color=_BLUE,  alpha=0.9)
    b2 = ax.bar(x + w/2, df.iloc[1][metrics], w, label=df.index[1], color=_GREEN, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison on ExDark Test Set', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    return _save(fig, 'model_comparison', save)


def plot_enhancement_impact(impact: dict, save: bool = True) -> plt.Figure:
    """Before/after chart for mAP50, precision, recall with improvement labels."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    metrics = [
        ('mAP50',     impact['mAP50_before'],     impact['mAP50_after'],     impact['mAP50_improvement_pct']),
        ('Precision', impact['precision_before'],  impact['precision_after'],  impact['precision_improvement_pct']),
        ('Recall',    impact['recall_before'],     impact['recall_after'],     impact['recall_improvement_pct']),
    ]
    for ax, (name, before, after, pct) in zip(axes, metrics):
        bars = ax.bar(['No enhance', 'CLAHE+Gamma'], [before, after],
                      color=[_RED, _GREEN], alpha=0.85, width=0.5)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylim(0, min(1.0, max(before, after) * 1.3))
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
        color = _GREEN if pct > 0 else _RED
        ax.text(0.5, 0.92, f'{"+%0.1f" % pct if pct > 0 else "%0.1f" % pct}%',
                transform=ax.transAxes, ha='center', fontsize=13,
                fontweight='bold', color=color)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Enhancement Impact: CLAHE + Gamma Correction', fontsize=13, fontweight='bold')
    return _save(fig, 'enhancement_impact', save)


def plot_fps_comparison(fps_data: dict, save: bool = True) -> plt.Figure:
    """Bar chart: FPS for each model variant."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_BLUE if 'nano' in k.lower() or 'n' in k.lower() else _PURPLE for k in fps_data]
    bars = ax.bar(list(fps_data.keys()), list(fps_data.values()), color=colors, alpha=0.9)
    ax.axhline(y=15, color=_RED, linestyle='--', linewidth=1.2,
               alpha=0.7, label='Min target: 15 FPS (CPU)')
    ax.axhline(y=30, color=_AMBER, linestyle='--', linewidth=1.2,
               alpha=0.7, label='Target: 30 FPS (GPU)')
    ax.set_ylabel('Frames per Second', fontsize=11)
    ax.set_title('Inference Speed — CPU vs GPU', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return _save(fig, 'fps_comparison', save)


def plot_class_distribution(class_counts: dict, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of detected objects by class."""
    if not class_counts:
        return None
    sorted_pairs = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_pairs)
    fig, ax = plt.subplots(figsize=(9, max(5, len(classes) * 0.55)))
    ax.barh(classes, counts, color=_PURPLE, alpha=0.85)
    ax.set_xlabel('Detection count', fontsize=11)
    ax.set_title('Objects Detected by Class', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    return _save(fig, 'class_distribution', save)


def plot_confidence_distribution(confidences: list, save: bool = True) -> plt.Figure:
    """Histogram of confidence scores across all detections."""
    if not confidences:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=20, color=_BLUE, alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(confidences), color=_RED, linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(confidences):.2f}')
    ax.set_xlabel('Confidence Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Detection Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return _save(fig, 'confidence_distribution', save)
