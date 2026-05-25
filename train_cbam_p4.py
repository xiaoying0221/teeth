#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口腔影像异常区域检测 - YOLOv8s + CBAM(P4)
exp10: 仅在 Backbone P4 输出后插入 CBAM，从官方预训练权重开始训练 250 轮
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML  = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
MODEL_YAML = Path(r"d:\文档\毕业设计\teeth\yolov8s-cbam-p4.yaml")
OUT_DIR    = Path(r"d:\文档\毕业设计\teeth\runs")

# ============================================================
# 训练配置
# ============================================================
EPOCHS  = 250
IMGSZ   = 640
BATCH   = 16
DEVICE  = "0"
WORKERS = 4
NAME    = "exp10_yolov8s_cbam_p4_250ep"


def main():
    print("=" * 60)
    print("口腔影像异常区域检测 - YOLOv8s + CBAM(P4)")
    print(f"模型结构: {MODEL_YAML}")
    print(f"Epochs: {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print(f"设备: GPU {DEVICE}")
    print("=" * 60)

    model = YOLO(str(MODEL_YAML))

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
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"结果保存在: {OUT_DIR / NAME}")
    print("=" * 60)

    best_pt = OUT_DIR / NAME / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    print("\n[验证] 使用 best.pt 在 val 集上评估...")
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(OUT_DIR),
        name=NAME + "_val_eval",
        exist_ok=True,
        verbose=True,
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    print("\n[测试] 使用 best.pt 在 test 集上评估...")
    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(OUT_DIR),
        name=NAME + "_test_eval",
        exist_ok=True,
        verbose=True,
    )
    print(f"Test Precision: {test_metrics.box.mp:.4f}")
    print(f"Test Recall:    {test_metrics.box.mr:.4f}")
    print(f"Test mAP50:     {test_metrics.box.map50:.4f}")
    print(f"Test mAP50-95:  {test_metrics.box.map:.4f}")

    print("\n[推理] 在 test 集上生成可视化结果...")
    test_images = str(Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test"))
    best_model.predict(
        source=test_images,
        save=True,
        save_txt=True,
        conf=0.25,
        project=str(OUT_DIR),
        name=NAME + "_predict",
        exist_ok=True,
    )
    print(f"可视化结果保存在: {OUT_DIR / (NAME + '_predict')}")


if __name__ == "__main__":
    main()
