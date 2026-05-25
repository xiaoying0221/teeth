#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口腔影像异常区域检测 - YOLOv8s 续训至 250 轮
从 exp3 的 last.pt 继续训练
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR   = Path(r"d:\文档\毕业设计\teeth\runs")

# ============================================================
# 训练配置
# ============================================================
MODEL     = str(OUT_DIR / "exp3_yolov8s_100ep/weights/last.pt")  # 从 exp3 的第100轮继续
EPOCHS    = 250            # 总轮数（从第101轮继续到250轮）
IMGSZ     = 640
BATCH     = 16
DEVICE    = "0"            # GPU
WORKERS   = 4
PROJECT   = str(OUT_DIR)
NAME      = "exp4_yolov8s_250ep"


def main():
    print("=" * 60)
    print("口腔影像异常区域检测 - YOLOv8s 续训至 250 轮")
    print(f"起始权重: {MODEL}")
    print(f"Epochs: {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print(f"设备: GPU {DEVICE}")
    print("=" * 60)

    # 加载 exp3 的最后权重
    model = YOLO(MODEL)

    # 继续训练
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
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

    # 验证集评估
    print("\n[验证] 在 val 集上评估...")
    metrics = model.val()
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # 测试集推理
    print("\n[推理] 在 test 集上生成可视化结果...")
    test_images = str(Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test"))
    model.predict(
        source=test_images,
        save=True,
        save_txt=True,
        conf=0.25,
        project=PROJECT,
        name=NAME + "_predict",
        exist_ok=True,
    )
    print(f"可视化结果保存在: {OUT_DIR / (NAME + '_predict')}")


if __name__ == "__main__":
    main()
