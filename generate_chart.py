# generate_chart.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create charts directory
Path('results/charts').mkdir(parents=True, exist_ok=True)

# Load evaluation results
with open('results/metrics/eval_results.json', 'r') as f:
    data = json.load(f)

# Extract metrics (including mAP50-95)
metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
yolo_before = [
    data['yolov8n']['mAP50'],
    data['yolov8n']['mAP50-95'],
    data['yolov8n']['precision'],
    data['yolov8n']['recall']
]
yolo_after = [
    data['best_ft']['mAP50'],
    data['best_ft']['mAP50-95'],
    data['best_ft']['precision'],
    data['best_ft']['recall']
]

# Create figure with adjusted size for 4 metrics
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, yolo_before, width, label='YOLOv8n (COCO pretrained)',
                color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, yolo_after, width, label='Fine-tuned on ExDark',
                color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.8)

# Customization
ax.set_ylabel('Performance Score', fontsize=12, fontweight='semibold')
ax.set_xlabel('Metrics', fontsize=12, fontweight='semibold')
ax.set_title('Model Comparison on ExDark Test Set\n+146,288% mAP50 Improvement after Fine-tuning', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper left', fontsize=10, frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=8.5, fontweight='bold', color='black', rotation=0)

# Add improvement annotation
improvement = data['impact']['mAP50_improvement_pct']
ax.text(0.5, 0.93, f' mAP50 Improvement: +{improvement:.0f}%', 
        transform=ax.transAxes, ha='center', fontsize=11, 
        fontweight='bold', color='#A23B72',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#A23B72'))

# Add FPS and mAP50-95 improvement as text annotation
fps_improve = ((data['fps_best'] - data['fps_nano']) / data['fps_nano'] * 100)
map95_improve = ((data['best_ft']['mAP50-95'] - data['yolov8n']['mAP50-95']) / data['yolov8n']['mAP50-95'] * 100)

info_text = f"FPS: {data['fps_nano']:.1f} → {data['fps_best']:.1f} (+{fps_improve:.0f}%)\nmAP50-95: {data['yolov8n']['mAP50-95']:.4f} → {data['best_ft']['mAP50-95']:.3f} (+{map95_improve:.0f}%)"

ax.text(0.97, 0.05, info_text, transform=ax.transAxes, ha='right', 
        fontsize=8, style='italic', verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig('results/charts/model_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n" + "="*65)
print(" Chart saved to: results/charts/model_comparison.png")
print("="*65)
print(f"\n RESULTS SUMMARY:")
print(f"   mAP50:      {data['yolov8n']['mAP50']:.4f} → {data['best_ft']['mAP50']:.3f} (+{improvement:.0f}%)")
print(f"   mAP50-95:   {data['yolov8n']['mAP50-95']:.4f} → {data['best_ft']['mAP50-95']:.3f} (+{map95_improve:.0f}%)")
print(f"   Precision:  {data['yolov8n']['precision']:.4f} → {data['best_ft']['precision']:.3f}")
print(f"   Recall:     {data['yolov8n']['recall']:.4f} → {data['best_ft']['recall']:.3f}")
print(f"   FPS:        {data['fps_nano']:.1f} → {data['fps_best']:.1f} (+{fps_improve:.0f}%)")
print("="*65)