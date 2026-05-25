# -*- coding: utf-8 -*-
"""
对比基线模型与改进模型关键训练曲线。

默认会尝试读取：
- baseline: runs/exp4_yolov8s_250ep/results.csv
- improved: runs/exp10_yolov8s_cbam_p4_250ep/results.csv

如果 results.csv 不存在，且对应目录下存在 weights/last.pt，脚本会尝试从 checkpoint 恢复 CSV。
输出图片：runs/compare_baseline_improved_curves.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None

BASE = Path(r"d:\文档\毕业设计\teeth")
RUNS = BASE / "runs"

EXPERIMENTS = {
    "baseline": {
        "title": "Baseline YOLOv8s",
        "results": RUNS / "exp4_yolov8s_250ep" / "results.csv",
        "checkpoint": RUNS / "exp4_yolov8s_250ep" / "weights" / "last.pt",
        "color": "#1f77b4",
    },
    "improved": {
        "title": "Improved YOLOv8s + CBAM(P4)",
        "results": RUNS / "exp10_yolov8s_cbam_p4_250ep" / "results.csv",
        "checkpoint": RUNS / "exp10_yolov8s_cbam_p4_250ep" / "weights" / "last.pt",
        "color": "#d62728",
    },
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_from_checkpoint(ckpt_path: Path, csv_path: Path) -> bool:
    if torch is None or not ckpt_path.exists():
        return False

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    tr = ckpt.get("train_results")
    if not tr:
        return False

    keys = list(tr.keys())
    rows = len(tr.get("epoch", []))
    if rows == 0:
        return False

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(rows):
            w.writerow([tr[k][i] for k in keys])
    return True


def ensure_results(path: Path, ckpt_path: Path) -> Optional[Path]:
    if path.exists():
        return path
    if write_csv_from_checkpoint(ckpt_path, path):
        print(f"Recovered CSV: {path}")
        return path
    print(f"Missing results and checkpoint: {path}")
    return None


def get_col(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


series = {}
for alias, cfg in EXPERIMENTS.items():
    csv_path = ensure_results(cfg["results"], cfg["checkpoint"])
    if csv_path is None:
        series[alias] = None
        continue
    rows = read_csv(csv_path)
    series[alias] = {
        "epoch": [int(r["epoch"]) for r in rows],
        "mAP50": get_col(rows, "metrics/mAP50(B)"),
        "mAP5095": get_col(rows, "metrics/mAP50-95(B)"),
        "val_loss": get_col(rows, "val/box_loss"),
    }

if series["baseline"] is None or series["improved"] is None:
    raise SystemExit("Baseline or improved experiment data is missing. Please ensure results.csv exists or last.pt is available.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)

plots = [
    ("mAP50", "mAP@50", "metrics/mAP50(B)"),
    ("mAP5095", "mAP@50-95", "metrics/mAP50-95(B)"),
    ("val_loss", "Val Loss", "val/box_loss"),
]

for ax, (key, ylabel, _) in zip(axes, plots):
    for alias in ("baseline", "improved"):
        data = series[alias]
        ax.plot(
            data["epoch"],
            data[key],
            label=EXPERIMENTS[alias]["title"],
            linewidth=2.2,
            color=EXPERIMENTS[alias]["color"],
        )
    ax.set_title(ylabel)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)

fig.suptitle("Baseline vs Improved Model Training Curves", fontsize=15, y=1.02)
fig.tight_layout()

out = RUNS / "compare_baseline_improved_curves.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved figure to {out}")
plt.show()
