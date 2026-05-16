"""
Metric computation and comparison utilities.
All output used directly in report tables.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def pct_change(before: float, after: float) -> float:
    """Percentage change from before to after. Returns 0 if before == 0."""
    if before == 0:
        return 0.0
    return round(((after - before) / before) * 100, 2)


def compute_enhancement_impact(metrics_raw: dict, metrics_enhanced: dict) -> dict:
    """
    Compute improvement from enhancement. THIS feeds the >15% requirement.

    Args:
        metrics_raw:      output of evaluate_on_dataset() without enhancement
        metrics_enhanced: output of evaluate_on_dataset() with enhancement preprocessed
    Returns:
        Dict with before/after values and percentage improvements
    """
    return {
        'mAP50_before':           metrics_raw['mAP50'],
        'mAP50_after':            metrics_enhanced['mAP50'],
        'mAP50_improvement_pct':  pct_change(metrics_raw['mAP50'], metrics_enhanced['mAP50']),

        'precision_before':       metrics_raw['precision'],
        'precision_after':        metrics_enhanced['precision'],
        'precision_improvement_pct': pct_change(metrics_raw['precision'], metrics_enhanced['precision']),

        'recall_before':          metrics_raw['recall'],
        'recall_after':           metrics_enhanced['recall'],
        'recall_improvement_pct': pct_change(metrics_raw['recall'], metrics_enhanced['recall']),
    }


def build_model_comparison_df(metrics_n: dict, metrics_s: dict) -> pd.DataFrame:
    """
    Build a DataFrame comparing YOLOv8n vs fine-tuned best.pt.
    Columns: mAP50, mAP50_95, precision, recall
    """
    rows = []
    for m in [metrics_n, metrics_s]:
        rows.append({
            'Model':     m['model'],
            'mAP50':     m['mAP50'],
            'mAP50-95':  m['mAP50_95'],
            'Precision': m['precision'],
            'Recall':    m['recall'],
        })
    return pd.DataFrame(rows).set_index('Model')


def save_results(results: dict, path: str = 'results/metrics/eval_results.json'):
    """Persist all evaluation results to JSON for dashboard + report."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    results['generated_at'] = datetime.now().isoformat()
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path: str = 'results/metrics/eval_results.json') -> dict:
    """Load previously saved evaluation results."""
    with open(path) as f:
        return json.load(f)
