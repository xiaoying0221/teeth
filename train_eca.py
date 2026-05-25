#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口腔影像异常区域检测 - YOLOv8s + ECA 注意力机制
exp9: 在 Backbone P3/P4/P5 输出后插入 ECA，从官方预训练权重开始训练 250 轮
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML  = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
MODEL_YAML = Path(r"d:\文档\毕业设计\teeth\yolov8s-eca.yaml")
OUT_DIR    = Path(r"d:\文档\毕业设计\teeth\runs")

# ============================================================
# 训练配置
# ============================================================
EPOCHS  = 250
IMGSZ   = 640
BATCH   = 16
DEVICE  = "0"
WORKERS = 4
NAME    = "exp9_yolov8s_eca_250ep"


def main():
    print("=" * 60)
    print("口腔影像异常区域检测 - YOLOv8s + ECA")
    print(f"模型结构: {MODEL_YAML}")
    print(f"Epochs: {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print(f"设备: GPU {DEVICE}")
    print("=" * 60)

    # 用自定义 YAML 初始化模型结构（预训练权重会自动匹配可用层）
    model = YOLO(str(MODEL_YAML))

    # 训练
    # 这里显式指定 SGD，避免 optimizer=auto 选到 MuSGD 后与 ECA 的 Conv1d 参数不兼容
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
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # 与 exp5/exp8 保持一致，确保公平对比
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
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    # test 集推理可视化
    print("\n[推理] 在 test 集上生成可视化结果...")
    test_images = str(Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test"))
    model.predict(
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
