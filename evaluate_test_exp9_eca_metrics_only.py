#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp9 测试集精简评估脚本
只输出四个指标：Precision、Recall、mAP50、mAP50-95
不做推理可视化
"""

from pathlib import Path
from ultralytics import YOLO

BEST_PT = Path(r"d:\文档\毕业设计\teeth\runs\exp9_yolov8s_eca_250ep\weights\best.pt")
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR = Path(r"d:\文档\毕业设计\teeth\runs")


def main():
    print("=" * 60)
    print("exp9 测试集精简评估")
    print(f"权重: {BEST_PT}")
    print(f"数据: {DATA_YAML}")
    print("=" * 60)

    if not BEST_PT.exists():
        print("未找到 best.pt，请先确认 exp9 已训练完成。")
        return

    model = YOLO(str(BEST_PT))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        project=str(OUT_DIR),
        name="exp9_test_eval_metrics_only",
        exist_ok=True,
        verbose=False,
    )

    print("\n" + "=" * 60)
    print("exp9 测试集结果")
    print("=" * 60)
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
