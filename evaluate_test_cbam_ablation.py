#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试 exp10 / exp11 / exp12 在 test 集上的效果
输出：Precision、Recall、mAP50、mAP50-95
并保存 test 集推理可视化结果
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
OUT_DIR   = Path(r"d:\文档\毕业设计\teeth\runs")
TEST_IMG  = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test")

EXPERIMENTS = [
    (
        "exp10 YOLOv8s+CBAM(P4)",
        Path(r"d:\文档\毕业设计\teeth\runs\exp10_yolov8s_cbam_p4_250ep\weights\best.pt"),
        "exp10_test_eval",
        "exp10_test_predict",
    ),
    (
        "exp11 YOLOv8s+CBAM(P4+P5)",
        Path(r"d:\文档\毕业设计\teeth\runs\exp11_yolov8s_cbam_p4p5_250ep\weights\best.pt"),
        "exp11_test_eval",
        "exp11_test_predict",
    ),
    (
        "exp12 YOLOv8s+CBAM(Neck)",
        Path(r"d:\文档\毕业设计\teeth\runs\exp12_yolov8s_cbam_neck_250ep\weights\best.pt"),
        "exp12_test_eval",
        "exp12_test_predict",
    ),
]


def evaluate_one(title, best_pt, eval_name, pred_name):
    print("\n" + "=" * 70)
    print(f"测试集评估 - {title}")
    print(f"权重: {best_pt}")
    print("=" * 70)

    if not best_pt.exists():
        print("权重不存在，跳过。")
        return None

    model = YOLO(str(best_pt))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        project=str(OUT_DIR),
        name=eval_name,
        exist_ok=True,
        verbose=True,
    )

    print("\n[测试集结果]")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    print("\n[推理] 在 test 集上生成可视化结果...")
    model.predict(
        source=str(TEST_IMG),
        save=True,
        save_txt=True,
        conf=0.25,
        project=str(OUT_DIR),
        name=pred_name,
        exist_ok=True,
    )
    print(f"可视化结果保存在: {OUT_DIR / pred_name}")

    return {
        "title": title,
        "p": metrics.box.mp,
        "r": metrics.box.mr,
        "m50": metrics.box.map50,
        "m95": metrics.box.map,
    }


def main():
    summary = []
    for title, best_pt, eval_name, pred_name in EXPERIMENTS:
        result = evaluate_one(title, best_pt, eval_name, pred_name)
        if result is not None:
            summary.append(result)

    print("\n" + "=" * 70)
    print("exp10 / exp11 / exp12 测试集汇总")
    print("=" * 70)
    print('{:24s} {:>10s} {:>10s} {:>10s} {:>12s}'.format(
        '模型', 'P', 'R', 'mAP50', 'mAP50-95'))
    print('-' * 70)
    for s in summary:
        print('{:24s} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}'.format(
            s['title'], s['p'], s['r'], s['m50'], s['m95']))
    print('=' * 70)


if __name__ == "__main__":
    main()
