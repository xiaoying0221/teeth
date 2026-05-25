# -*- coding: utf-8 -*-
"""Find two kinds of typical comparison cases for exp8 vs exp10.

Case A: false positives in complex background
- exp10 predicts at least one box
- exp8 predicts no box
- no ground-truth object nearby / ideally GT count is 0

Case B: both detect, but exp8 is more precise
- both exp8 and exp10 predict boxes
- exp8 boxes are closer to GT than exp10 boxes (higher IoU / better match)

This script uses YOLO txt labels from prediction folders and ground-truth labels.
It outputs two candidate lists:
- typical_fp_candidates.txt / csv
- typical_precision_candidates.txt / csv

Expected folders:
- runs/exp8_test_predict/labels
- runs/exp10_test_predict/labels OR runs/exp10_yolov8s_cbam_p4_250ep_predict/labels
- dataset_yolo/labels/test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional, List, Tuple

BASE = Path(r"d:\文档\毕业设计\teeth")
EXP8_PRED = BASE / "runs" / "exp8_test_predict" / "labels"
EXP10_PRED_CANDIDATES = [
    BASE / "runs" / "exp10_test_predict" / "labels",
    BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep_predict" / "labels",
]
GT_LABELS = BASE / "dataset_yolo" / "labels" / "test"

OUT_FP_CSV = BASE / "typical_fp_candidates.csv"
OUT_FP_TXT = BASE / "typical_fp_candidates.txt"
OUT_PREC_CSV = BASE / "typical_precision_candidates.csv"
OUT_PREC_TXT = BASE / "typical_precision_candidates.txt"

IOU_THRESHOLD = 0.30
TOP_N = 20


@dataclass
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def read_boxes(txt_path: Path) -> List[Box]:
    if not txt_path.exists():
        return []
    boxes: List[Box] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:5])
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            boxes.append(Box(cls_id, x1, y1, x2, y2))
    return boxes


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def best_pred_iou(pred: Box, gts: List[Box]) -> float:
    if not gts:
        return 0.0
    return max(iou(pred, gt) for gt in gts)


def load_pred_dir(candidates: List[Path]) -> Path:
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError("Cannot find exp10 prediction labels folder")


def stems_from_dir(d: Path) -> set[str]:
    return {p.stem for p in d.glob("*.txt")}


@dataclass
class FpRow:
    image: str
    gt_count: Optional[int]
    exp8_count: int
    exp10_count: int
    score: float


@dataclass
class PrecRow:
    image: str
    gt_count: Optional[int]
    exp8_count: int
    exp10_count: int
    exp8_best_iou: float
    exp10_best_iou: float
    score: float


def find_fp_candidates(exp10_dir: Path) -> List[FpRow]:
    rows: List[FpRow] = []
    all_stems = sorted(stems_from_dir(EXP8_PRED) | stems_from_dir(exp10_dir))
    for stem in all_stems:
        exp8_boxes = read_boxes(EXP8_PRED / f"{stem}.txt")
        exp10_boxes = read_boxes(exp10_dir / f"{stem}.txt")
        gt_boxes = read_boxes(GT_LABELS / f"{stem}.txt")
        gt_count = len(gt_boxes) if gt_boxes else 0

        # Case A: exp10 has detections, exp8 has none; GT preferably empty.
        if len(exp10_boxes) == 0 or len(exp8_boxes) != 0:
            continue

        # Prefer true false positives in complex background: no GT objects.
        # If there are GTs, allow only if exp10 boxes have no reasonable overlap.
        if gt_count == 0:
            score = float(len(exp10_boxes)) + 2.0
            rows.append(FpRow(stem, gt_count, len(exp8_boxes), len(exp10_boxes), score))
            continue

        max_iou = max((best_pred_iou(b, gt_boxes) for b in exp10_boxes), default=0.0)
        if max_iou < IOU_THRESHOLD:
            score = float(len(exp10_boxes)) + (IOU_THRESHOLD - max_iou)
            rows.append(FpRow(stem, gt_count, len(exp8_boxes), len(exp10_boxes), score))

    rows.sort(key=lambda r: (r.score, r.exp10_count, -(r.gt_count or 0)), reverse=True)
    return rows


def avg_best_iou(pred_boxes: List[Box], gt_boxes: List[Box]) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    vals = [best_pred_iou(pb, gt_boxes) for pb in pred_boxes]
    return sum(vals) / len(vals)


def find_precision_candidates(exp10_dir: Path) -> List[PrecRow]:
    rows: List[PrecRow] = []
    all_stems = sorted(stems_from_dir(EXP8_PRED) | stems_from_dir(exp10_dir))
    for stem in all_stems:
        exp8_boxes = read_boxes(EXP8_PRED / f"{stem}.txt")
        exp10_boxes = read_boxes(exp10_dir / f"{stem}.txt")
        gt_boxes = read_boxes(GT_LABELS / f"{stem}.txt")
        gt_count = len(gt_boxes) if gt_boxes else 0

        # Case B: both detect something and GT exists.
        if not exp8_boxes or not exp10_boxes or gt_count == 0:
            continue

        exp8_iou = avg_best_iou(exp8_boxes, gt_boxes)
        exp10_iou = avg_best_iou(exp10_boxes, gt_boxes)

        # exp8 should be more precise
        if exp8_iou <= exp10_iou:
            continue

        # score combines precision gap and count similarity to GT
        score = (exp8_iou - exp10_iou) * 10.0
        score += max(0.0, abs(len(exp10_boxes) - gt_count) - abs(len(exp8_boxes) - gt_count))
        rows.append(PrecRow(stem, gt_count, len(exp8_boxes), len(exp10_boxes), exp8_iou, exp10_iou, score))

    rows.sort(key=lambda r: (r.score, r.exp8_best_iou - r.exp10_best_iou, -(r.gt_count or 0)), reverse=True)
    return rows


def write_fp(rows: List[FpRow]) -> None:
    with OUT_FP_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_count", "exp8_count", "exp10_count", "score"])
        for r in rows:
            w.writerow([r.image, r.gt_count, r.exp8_count, r.exp10_count, f"{r.score:.3f}"])

    with OUT_FP_TXT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows[:TOP_N], start=1):
            f.write(f"{i:03d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} | score={r.score:.3f}\n")


def write_prec(rows: List[PrecRow]) -> None:
    with OUT_PREC_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_count", "exp8_count", "exp10_count", "exp8_best_iou", "exp10_best_iou", "score"])
        for r in rows:
            w.writerow([r.image, r.gt_count, r.exp8_count, r.exp10_count, f"{r.exp8_best_iou:.4f}", f"{r.exp10_best_iou:.4f}", f"{r.score:.3f}"])

    with OUT_PREC_TXT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows[:TOP_N], start=1):
            f.write(
                f"{i:03d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} "
                f"| exp8_iou={r.exp8_best_iou:.3f} | exp10_iou={r.exp10_best_iou:.3f} | score={r.score:.3f}\n"
            )


def main() -> None:
    if not EXP8_PRED.exists():
        raise FileNotFoundError(f"Missing exp8 prediction labels folder: {EXP8_PRED}")
    exp10_dir = load_pred_dir(EXP10_PRED_CANDIDATES)
    if not GT_LABELS.exists():
        raise FileNotFoundError(f"Missing GT labels folder: {GT_LABELS}")

    fp_rows = find_fp_candidates(exp10_dir)
    prec_rows = find_precision_candidates(exp10_dir)

    write_fp(fp_rows)
    write_prec(prec_rows)

    print("[A] Complex-background false positives (exp10 hits, exp8 misses)")
    print("Found:", len(fp_rows))
    print("TXT:", OUT_FP_TXT)
    print("CSV:", OUT_FP_CSV)
    for i, r in enumerate(fp_rows[:10], start=1):
        print(f"{i:02d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} | score={r.score:.3f}")

    print()
    print("[B] Both detect, but exp8 is more precise")
    print("Found:", len(prec_rows))
    print("TXT:", OUT_PREC_TXT)
    print("CSV:", OUT_PREC_CSV)
    for i, r in enumerate(prec_rows[:10], start=1):
        print(
            f"{i:02d}. {r.image} | GT={r.gt_count} | exp8={r.exp8_count} | exp10={r.exp10_count} "
            f"| exp8_iou={r.exp8_best_iou:.3f} | exp10_iou={r.exp10_best_iou:.3f} | score={r.score:.3f}"
        )


if __name__ == "__main__":
    main()
