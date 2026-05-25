# -*- coding: utf-8 -*-
"""Find comparison images where exp8 detects more than exp10.

This script compares prediction txt files from two YOLO prediction folders and
optionally matches them against ground-truth labels to find typical cases:
- exp8 detects more objects than exp10
- exp10 misses objects that exp8 detects
- exp8 has a prediction count closer to ground truth

Usage example:
    python find_typical_compare_images_exp8_more.py

Outputs:
- comparison_candidates_exp8_more.csv
- comparison_candidates_exp8_more.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional

BASE = Path(r"d:\文档\毕业设计\teeth")
EXP8_PRED = BASE / "runs" / "exp8_test_predict" / "labels"
EXP10_PRED = BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep_predict" / "labels"
GT_LABELS = BASE / "dataset_yolo" / "labels" / "test"
OUT_CSV = BASE / "comparison_candidates_exp8_more.csv"
OUT_TXT = BASE / "comparison_candidates_exp8_more.txt"

# If you also want to require exp8 to be closer to GT than exp10, keep True.
REQUIRE_EXP8_BETTER_THAN_EXP10 = True


@dataclass
class Row:
    image: str
    gt_count: Optional[int]
    exp8_count: int
    exp10_count: int
    delta_exp8_minus_exp10: int
    delta_exp8_minus_gt: Optional[int]
    delta_exp10_minus_gt: Optional[int]
    score: float


def count_boxes(txt_path: Path) -> int:
    if not txt_path.exists():
        return 0
    with txt_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_candidates() -> list[Row]:
    rows: list[Row] = []
    exp8_files = {p.stem for p in EXP8_PRED.glob("*.txt")}
    exp10_files = {p.stem for p in EXP10_PRED.glob("*.txt")}
    all_stems = sorted(exp8_files | exp10_files)

    for stem in all_stems:
        exp8_count = count_boxes(EXP8_PRED / f"{stem}.txt")
        exp10_count = count_boxes(EXP10_PRED / f"{stem}.txt")
        gt_path = GT_LABELS / f"{stem}.txt"
        gt_count = count_boxes(gt_path) if gt_path.exists() else None

        delta = exp8_count - exp10_count
        delta_exp8_gt = None if gt_count is None else exp8_count - gt_count
        delta_exp10_gt = None if gt_count is None else exp10_count - gt_count

        # score: prefer exp8 > exp10, and exp8 closer to GT count
        score = float(delta)
        if gt_count is not None:
            score += max(0.0, abs(delta_exp10_gt) - abs(delta_exp8_gt))
            score += 0.1 * (gt_count - exp10_count)

        if delta <= 0:
            continue
        if REQUIRE_EXP8_BETTER_THAN_EXP10 and gt_count is not None:
            if abs(delta_exp8_gt) > abs(delta_exp10_gt):
                continue

        rows.append(
            Row(
                image=stem,
                gt_count=gt_count,
                exp8_count=exp8_count,
                exp10_count=exp10_count,
                delta_exp8_minus_exp10=delta,
                delta_exp8_minus_gt=delta_exp8_gt,
                delta_exp10_minus_gt=delta_exp10_gt,
                score=score,
            )
        )

    rows.sort(key=lambda r: (r.score, r.delta_exp8_minus_exp10, -(r.gt_count or 0)), reverse=True)
    return rows


def write_outputs(rows: list[Row]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "image",
            "gt_count",
            "exp8_count",
            "exp10_count",
            "delta_exp8_minus_exp10",
            "delta_exp8_minus_gt",
            "delta_exp10_minus_gt",
            "score",
        ])
        for r in rows:
            w.writerow([
                r.image,
                r.gt_count if r.gt_count is not None else "",
                r.exp8_count,
                r.exp10_count,
                r.delta_exp8_minus_exp10,
                r.delta_exp8_minus_gt if r.delta_exp8_minus_gt is not None else "",
                r.delta_exp10_minus_gt if r.delta_exp10_minus_gt is not None else "",
                f"{r.score:.3f}",
            ])

    with OUT_TXT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            f.write(
                f"{i:03d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} "
                f"| delta={r.delta_exp8_minus_exp10} | score={r.score:.3f}\n"
            )


def main() -> None:
    if not EXP8_PRED.exists():
        raise FileNotFoundError(f"Missing exp8 prediction labels folder: {EXP8_PRED}")
    if not EXP10_PRED.exists():
        raise FileNotFoundError(f"Missing exp10 prediction labels folder: {EXP10_PRED}")

    rows = load_candidates()
    write_outputs(rows)

    print("Found candidates:", len(rows))
    print("CSV:", OUT_CSV)
    print("TXT:", OUT_TXT)
    print("Top 10:")
    for i, r in enumerate(rows[:10], start=1):
        print(
            f"{i:02d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} "
            f"| delta={r.delta_exp8_minus_exp10} | score={r.score:.3f}"
        )


if __name__ == "__main__":
    main()
