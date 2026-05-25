#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""筛选 exp8 vs exp10 典型对比图。

目标：自动遍历测试集预测结果，找出“exp10 检出更多、exp8 漏检”的图片。

用法：
1) 先分别对 test 集运行 exp8 和 exp10 的预测，确保生成对应的 labels 文本。
2) 再运行本脚本，它会比对每张图的预测框数量，输出候选列表。

默认假设目录结构类似：
- runs/exp8_yolov8s_250ep_scratch_predict/labels
- runs/exp10_yolov8s_cbam_p4_250ep_predict/labels
- dataset_yolo/labels/test

如果你的预测输出目录不同，修改下面的路径即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json

BASE = Path(r"d:\文档\毕业设计\teeth")
GT_DIR = BASE / "dataset_yolo" / "labels" / "test"
EXP8_DIR = BASE / "runs" / "exp8_yolov8s_250ep_scratch_predict" / "labels"
EXP10_DIR = BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep_predict" / "labels"
OUT_CSV = BASE / "exp8_vs_exp10_典型对比图候选.csv"
OUT_JSON = BASE / "exp8_vs_exp10_典型对比图候选.json"


def read_yolo_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO txt label file.

    Returns list of (cls, x, y, w, h).
    """
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            items.append((cls, x, y, w, h))
    return items


def list_stems(dir_path: Path) -> Dict[str, Path]:
    mapping = {}
    if not dir_path.exists():
        return mapping
    for p in dir_path.glob("*.txt"):
        mapping[p.stem] = p
    return mapping


def main():
    if not GT_DIR.exists():
        raise FileNotFoundError(f"Ground truth dir not found: {GT_DIR}")
    if not EXP8_DIR.exists():
        raise FileNotFoundError(f"exp8 labels dir not found: {EXP8_DIR}")
    if not EXP10_DIR.exists():
        raise FileNotFoundError(f"exp10 labels dir not found: {EXP10_DIR}")

    gt_files = list_stems(GT_DIR)
    exp8_files = list_stems(EXP8_DIR)
    exp10_files = list_stems(EXP10_DIR)

    candidates = []

    for stem, gt_path in gt_files.items():
        gt = read_yolo_label_file(gt_path)
        pred8 = read_yolo_label_file(exp8_files.get(stem, Path()))
        pred10 = read_yolo_label_file(exp10_files.get(stem, Path()))

        gt_n = len(gt)
        p8_n = len(pred8)
        p10_n = len(pred10)

        # 典型候选：exp10 比 exp8 检出更多，且 exp8 存在漏检迹象
        # 条件可以按需调整：
        # 1) exp10 比 exp8 多 1 个及以上框
        # 2) exp10 的框数至少不低于 GT，或者 exp8 明显少于 GT
        if p10_n > p8_n and (p8_n < gt_n or p10_n >= gt_n):
            candidates.append({
                "image": stem,
                "gt_count": gt_n,
                "exp8_count": p8_n,
                "exp10_count": p10_n,
                "delta_exp10_minus_exp8": p10_n - p8_n,
            })

    # 优先排序：exp10 比 exp8 多得越多越靠前；其次看 exp8 漏得越多越靠前
    candidates.sort(key=lambda x: (x["delta_exp10_minus_exp8"], x["gt_count"] - x["exp8_count"]), reverse=True)

    # 输出 CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_count", "exp8_count", "exp10_count", "delta_exp10_minus_exp8"])
        writer.writeheader()
        writer.writerows(candidates)

    # 输出 JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    print("筛选完成")
    print(f"候选数量: {len(candidates)}")
    print(f"CSV:  {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print()
    print("前 20 个候选:")
    for item in candidates[:20]:
        print(
            f"{item['image']}: GT={item['gt_count']}, exp8={item['exp8_count']}, exp10={item['exp10_count']}, Δ={item['delta_exp10_minus_exp8']}"
        )


if __name__ == "__main__":
    main()
