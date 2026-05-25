#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp8: YOLOv8s（无CBAM），从官方预训练权重从零训练 250 轮
用途：与 exp5（YOLOv8s+CBAM，同样从零，250轮）做控制变量对比
唯一变量：有无 CBAM
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR   = Path(r"d:\文档\毕业设计\teeth\runs")
NAME      = "exp8_yolov8s_250ep_scratch"

# ============================================================
# 训练配置（与 exp5 完全一致，保证控制变量）
# ============================================================
EPOCHS  = 250
IMGSZ   = 640
BATCH   = 16
DEVICE  = "0"
WORKERS = 4


def main():
    print("=" * 60)
    print("exp8: YOLOv8s（无CBAM）从零训练 250 轮")
    print("对照实验：与 exp5（YOLOv8s+CBAM，从零250轮）控制变量对比")
    print(f"起始权重: yolov8s.pt（官方预训练）")
    print(f"Epochs: {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print("=" * 60)

    # 用官方 yolov8s.pt 从零开始（与 exp5 起点相同）
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUT_DIR),
        name=NAME,
        exist_ok=True,
        verbose=True,
        # 与 exp5 完全相同的数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\n" + "=" * 60)
    print("训练完成！结果保存在:", OUT_DIR / NAME)
    print("=" * 60)

    print("\n[验证集评估]")
    metrics = model.val()
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n[测试集评估]")
    test_metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(OUT_DIR),
        name=NAME + "_test_eval",
        exist_ok=True,
    )
    print("\n测试集结果:")
    print(f"  Precision:  {test_metrics.box.mp:.4f}")
    print(f"  Recall:     {test_metrics.box.mr:.4f}")
    print(f"  mAP50:      {test_metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {test_metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
