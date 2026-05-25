#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集评估脚本
使用 exp6_yolov8s_cbam_transfer 的 best.pt 权重在测试集上评估
输出：mAP50、mAP50-95、Precision、Recall
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
BEST_PT   = Path(r"d:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\weights\best.pt")
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR   = Path(r"d:\文档\毕业设计\teeth\runs")


def main():
    print("=" * 60)
    print("测试集评估 - exp6 YOLOv8s+CBAM+迁移 best.pt")
    print(f"权重: {BEST_PT}")
    print(f"数据: {DATA_YAML}")
    print("=" * 60)

    model = YOLO(str(BEST_PT))

    # 在测试集上评估
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        project=str(OUT_DIR),
        name="exp6_test_eval",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("测试集评估结果 - exp6 YOLOv8s+CBAM+迁移")
    print("=" * 60)
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 60)

    # 同时做测试集推理（保存可视化结果）
    print("\n[推理] 在 test 集上生成可视化结果...")
    test_images = str(Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test"))
    model.predict(
        source=test_images,
        save=True,
        save_txt=True,
        conf=0.25,
        project=str(OUT_DIR),
        name="exp6_test_eval_predict",
        exist_ok=True,
    )
    print(f"可视化结果保存在: {OUT_DIR / 'exp6_test_eval_predict'}")


if __name__ == "__main__":
    main()
