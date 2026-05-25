# -*- coding: utf-8 -*-
"""Plot training curve comparison for exp8 vs exp10.

Baseline: exp8_yolov8s_250ep_scratch
Improved : exp10_yolov8s_cbam_p4_250ep

Outputs:
- exp8_vs_exp10_curves.png
- exp8_vs_exp10_curves.pdf
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt

BASE = Path(r"d:\文档\毕业设计\teeth")
EXP4 = BASE / "runs" / "exp8_yolov8s_250ep_scratch" / "results.csv"
EXP10 = BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep" / "results.csv"
OUT_PNG = BASE / "exp8_vs_exp10_curves_v2.png"
OUT_PDF = BASE / "exp8_vs_exp10_curves_v2.pdf"


def read_results(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("epoch", "").strip()]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def extract(rows):
    epochs = [int(r["epoch"]) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    map5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    val_box = [float(r["val/box_loss"]) for r in rows]
    val_cls = [float(r["val/cls_loss"]) for r in rows]
    val_dfl = [float(r["val/dfl_loss"]) for r in rows]
    val_loss = [a + b + c for a, b, c in zip(val_box, val_cls, val_dfl)]
    return {
        "epochs": epochs,
        "map50": map50,
        "map5095": map5095,
        "val_loss": val_loss,
    }


def best_info(epochs, values):
    idx = max(range(len(values)), key=lambda i: values[i])
    return epochs[idx], values[idx]


def main():
    exp4 = extract(read_results(EXP4))
    exp10 = extract(read_results(EXP10))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
    fig.suptitle("基线模型与改进模型关键训练曲线对比", fontsize=16, fontweight="bold")

    # mAP@50
    ax = axes[0]
    ax.plot(exp4["epochs"], exp4["map50"], label="YOLOv8s", linewidth=2.2, color="#1f77b4")
    ax.plot(exp10["epochs"], exp10["map50"], label="YOLOv8s+CBAM(P4)", linewidth=2.2, color="#d62728")
    ax.set_title("mAP@50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # mAP@50-95
    ax = axes[1]
    ax.plot(exp4["epochs"], exp4["map5095"], label="YOLOv8s", linewidth=2.2, color="#1f77b4")
    ax.plot(exp10["epochs"], exp10["map5095"], label="YOLOv8s+CBAM(P4)", linewidth=2.2, color="#d62728")
    ax.set_title("mAP@50-95")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # val loss
    ax = axes[2]
    ax.plot(exp4["epochs"], exp4["val_loss"], label="YOLOv8s", linewidth=2.2, color="#1f77b4")
    ax.plot(exp10["epochs"], exp10["val_loss"], label="YOLOv8s+CBAM(P4)", linewidth=2.2, color="#d62728")
    ax.set_title("val loss (box + cls + dfl)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # annotations
    e4_50, v4_50 = best_info(exp4["epochs"], exp4["map50"])
    e10_50, v10_50 = best_info(exp10["epochs"], exp10["map50"])
    e4_95, v4_95 = best_info(exp4["epochs"], exp4["map5095"])
    e10_95, v10_95 = best_info(exp10["epochs"], exp10["map5095"])

    axes[0].annotate(f"best {v4_50:.3f}@{e4_50}", xy=(e4_50, v4_50), xytext=(10, -25),
                     textcoords="offset points", fontsize=9, color="#1f77b4")
    axes[0].annotate(f"best {v10_50:.3f}@{e10_50}", xy=(e10_50, v10_50), xytext=(10, 10),
                     textcoords="offset points", fontsize=9, color="#d62728")
    axes[1].annotate(f"best {v4_95:.3f}@{e4_95}", xy=(e4_95, v4_95), xytext=(10, -25),
                     textcoords="offset points", fontsize=9, color="#1f77b4")
    axes[1].annotate(f"best {v10_95:.3f}@{e10_95}", xy=(e10_95, v10_95), xytext=(10, 10),
                     textcoords="offset points", fontsize=9, color="#d62728")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
