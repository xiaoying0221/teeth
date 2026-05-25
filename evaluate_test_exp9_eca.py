#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集评估脚本
使用 exp9_yolov8s_eca_250ep 的 best.pt 权重在测试集上评估
输出：mAP50、mAP50-95、Precision、Recall
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
BEST_PT   = Path(r"d:\文档\毕业设计\teeth\runs\exp9_yolov8s_eca_250ep\weights\best.pt")
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR   = Path(r"d:\文档\毕业设计\teeth\runs")
TEST_IMG  = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test")


def main():
    print("=" * 60)
    print("测试集评估 - exp9 YOLOv8s+ECA")
    print(f"权重: {BEST_PT}")
    print(f"数据: {DATA_YAML}")
    print("=" * 60)

    if not BEST_PT.exists():
        print("未找到 best.pt，请先确认 exp9 已训练完成。")
        return

    model = YOLO(str(BEST_PT))

    # 在测试集上评估
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        project=str(OUT_DIR),
        name="exp9_test_eval",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("测试集评估结果 - exp9 YOLOv8s+ECA")
    print("=" * 60)
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 60)

    # 同时做测试集推理（保存可视化结果）
    print("\n[推理] 在 test 集上生成可视化结果...")
    model.predict(
        source=str(TEST_IMG),
        save=True,
        save_txt=True,
        conf=0.25,
        project=str(OUT_DIR),
        name="exp9_test_eval_predict",
        exist_ok=True,
    )
    print(f"可视化结果保存在: {OUT_DIR / 'exp9_test_eval_predict'}")


if __name__ == "__main__":
    main()
