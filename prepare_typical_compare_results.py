# -*- coding: utf-8 -*-
"""Copy top candidate images and build side-by-side comparison collages.

This script does three things for the exp8 vs exp10 comparison workflow:
1) Reads the candidate list from comparison_candidates_exp8_more.txt
2) Copies the top-N original test images into a separate folder
3) Creates left-right comparison collages using the saved prediction images

Expected folders:
- Original test images: dataset_yolo/images/test
- exp8 prediction images: runs/exp8_test_predict
- exp10 prediction images: runs/exp10_test_predict or runs/exp10_yolov8s_cbam_p4_250ep_predict

Outputs:
- selected_compare_images/
- compare_collages/
"""

from __future__ import annotations

import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"d:\文档\毕业设计\teeth")
CANDIDATE_TXT = BASE / "comparison_candidates_exp8_more.txt"
TEST_IMAGES = BASE / "dataset_yolo" / "images" / "test"
EXP8_PRED_DIR = BASE / "runs" / "exp8_test_predict"
EXP10_PRED_DIR_CANDIDATES = [
    BASE / "runs" / "exp10_test_predict",
    BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep_predict",
]
OUT_SELECTED = BASE / "selected_compare_images"
OUT_COLLAGES = BASE / "compare_collages"
TOP_N = 5


def read_candidate_stems(path: Path) -> list[str]:
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Example: 001. test_0187_(110) | GT=4 | exp8=2 | exp10=0 | delta=2 | score=4.400
        if ". " not in line:
            continue
        rest = line.split(". ", 1)[1]
        stem = rest.split(" | ", 1)[0].strip()
        stems.append(stem)
    return stems


def find_pred_dir() -> Path:
    for d in EXP10_PRED_DIR_CANDIDATES:
        if d.exists():
            return d
    raise FileNotFoundError("Cannot find exp10 prediction folder")


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def add_title(img: Image.Image, title: str) -> Image.Image:
    pad = 44
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("simhei.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, 10), title, fill="black", font=font)
    return canvas


def build_collage(exp8_img: Path, exp10_img: Path, out_path: Path, title: str) -> None:
    left = Image.open(exp8_img).convert("RGB")
    right = Image.open(exp10_img).convert("RGB")

    # resize to same height while keeping aspect ratio
    target_h = min(left.height, right.height)
    def resize_to_h(im: Image.Image, h: int) -> Image.Image:
        w = int(im.width * h / im.height)
        return im.resize((w, h))

    left = resize_to_h(left, target_h)
    right = resize_to_h(right, target_h)

    gutter = 24
    label_h = 44
    total_w = left.width + right.width + gutter
    total_h = target_h + label_h
    canvas = Image.new("RGB", (total_w, total_h), "white")
    canvas.paste(add_title(left, "YOLOv8s"), (0, 0))
    canvas.paste(add_title(right, "YOLOv8s+CBAM(P4)"), (left.width + gutter, 0))

    # optional small footer title
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("simhei.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, total_h - 22), title, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    if not CANDIDATE_TXT.exists():
        raise FileNotFoundError(f"Missing candidate list: {CANDIDATE_TXT}")
    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Missing test images folder: {TEST_IMAGES}")
    if not EXP8_PRED_DIR.exists():
        raise FileNotFoundError(f"Missing exp8 prediction folder: {EXP8_PRED_DIR}")

    exp10_pred_dir = find_pred_dir()

    stems = read_candidate_stems(CANDIDATE_TXT)[:TOP_N]
    if not stems:
        raise ValueError("No candidates found in candidate list")

    OUT_SELECTED.mkdir(parents=True, exist_ok=True)
    OUT_COLLAGES.mkdir(parents=True, exist_ok=True)

    # 1) copy original test images
    copied = []
    for stem in stems:
        # original test image may have .jpg/.png; search by stem
        src = next(TEST_IMAGES.glob(stem + ".*"), None)
        if src is None:
            print(f"[WARN] Missing original image for {stem}")
            continue
        dst = OUT_SELECTED / src.name
        safe_copy(src, dst)
        copied.append((stem, src, dst))

    # 2) create comparison collages from saved prediction images
    made = 0
    for stem, src_img, _ in copied:
        exp8_img = next(EXP8_PRED_DIR.glob(stem + ".*"), None)
        exp10_img = next(exp10_pred_dir.glob(stem + ".*"), None)
        if exp8_img is None or exp10_img is None:
            print(f"[WARN] Missing prediction image for {stem}: exp8={exp8_img}, exp10={exp10_img}")
            continue
        out = OUT_COLLAGES / f"{stem}_compare.jpg"
        build_collage(exp8_img, exp10_img, out, stem)
        made += 1

    print(f"Selected stems: {stems}")
    print(f"Copied originals to: {OUT_SELECTED}")
    print(f"Collages saved to: {OUT_COLLAGES}")
    print(f"Collages created: {made}")


if __name__ == "__main__":
    main()
