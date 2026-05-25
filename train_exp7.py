#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp7: YOLOv8s（无CBAM），从 exp4 best.pt 继续训练 200 轮
用途：与 exp6（YOLOv8s+CBAM，同样从 exp4 迁移，200轮）做控制变量对比
唯一变量：有无 CBAM
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML   = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
EXP4_WEIGHT = Path(r"d:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\weights\best.pt")
OUT_DIR     = Path(r"d:\文档\毕业设计\teeth\runs")
NAME        = "exp7_yolov8s_200ep_finetune"

# ============================================================
# 训练配置（与 exp6 完全一致，保证控制变量）
# ============================================================
EPOCHS  = 200
IMGSZ   = 640
BATCH   = 16
DEVICE  = "0"
WORKERS = 4


def main():
    print("=" * 60)
    print("exp7: YOLOv8s（无CBAM）从 exp4 best.pt 续训 200 轮")
    print("对照实验：与 exp6（YOLOv8s+CBAM，200轮）控制变量对比")
    print(f"起始权重: {EXP4_WEIGHT}")
    print(f"Epochs: {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print("=" * 60)

    # 直接加载 exp4 best.pt，结构是普通 YOLOv8s（无CBAM）
    model = YOLO(str(EXP4_WEIGHT))

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
        # 与 exp6 完全相同的数据增强
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

    print("\n[验证] 在 val 集上评估...")
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
