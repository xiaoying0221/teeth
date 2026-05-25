# -*- coding: utf-8 -*-
"""Build two kinds of typical comparison packs for exp8 vs exp10.

It reads two candidate lists and automatically:
1) selects top 3-5 cases for each type
2) copies original test images into separate folders
3) creates side-by-side collages

Types:
A. False positive under complex background
   - exp10 false positive
   - exp8 does not false positive

B. More accurate localization
   - both exp8 and exp10 detect
   - exp8 is closer to GT (by overlap score / count heuristic)

Expected input files from the candidate-finding scripts:
- typical_fp_candidates.txt
- typical_precision_candidates.txt

Expected folders:
- dataset_yolo/images/test
- runs/exp8_test_predict
- runs/exp10_test_predict or runs/exp10_yolov8s_cbam_p4_250ep_predict

Outputs:
- typical_compare_packs/
  - fp_case_images/
  - precision_case_images/
  - fp_case_collages/
  - precision_case_collages/
"""

from __future__ import annotations

from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"d:\文档\毕业设计\teeth")
TEST_IMAGES = BASE / "dataset_yolo" / "images" / "test"
EXP8_PRED_DIR = BASE / "runs" / "exp8_test_predict"
EXP10_PRED_DIR_CANDIDATES = [
    BASE / "runs" / "exp10_test_predict",
    BASE / "runs" / "exp10_yolov8s_cbam_p4_250ep_predict",
]

FP_TXT = BASE / "typical_fp_candidates.txt"
PREC_TXT = BASE / "typical_precision_candidates.txt"
OUT_ROOT = BASE / "typical_compare_packs"
TOP_N = 5


def read_stems(path: Path) -> list[str]:
    stems: list[str] = []
    if not path.exists():
        return stems
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ". " not in line:
            continue
        rest = line.split(". ", 1)[1]
        stem = rest.split(" | ", 1)[0].strip()
        stems.append(stem)
    return stems


def find_exp10_pred_dir() -> Path:
    for d in EXP10_PRED_DIR_CANDIDATES:
        if d.exists():
            return d
    raise FileNotFoundError("Cannot find exp10 prediction directory")


def first_match(folder: Path, stem: str) -> Path | None:
    candidates = list(folder.glob(stem + ".*"))
    return candidates[0] if candidates else None


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_font(size: int):
    for name in ["simhei.ttf", "msyh.ttc", "Microsoft YaHei.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_label_panel(img: Image.Image, title: str) -> Image.Image:
    pad = 40
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="black", font=load_font(20))
    return canvas


def build_collage(exp8_img: Path, exp10_img: Path, out_path: Path, title: str) -> None:
    left = Image.open(exp8_img).convert("RGB")
    right = Image.open(exp10_img).convert("RGB")
    h = min(left.height, right.height)

    def resize_to_h(im: Image.Image, target_h: int) -> Image.Image:
        w = int(im.width * target_h / im.height)
        return im.resize((w, target_h))

    left = resize_to_h(left, h)
    right = resize_to_h(right, h)
    left = make_label_panel(left, "YOLOv8s")
    right = make_label_panel(right, "YOLOv8s+CBAM(P4)")

    gutter = 24
    footer_h = 34
    canvas = Image.new("RGB", (left.width + right.width + gutter, left.height + footer_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gutter, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, canvas.height - 24), title, fill="black", font=load_font(18))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def process_case_list(name: str, stems: list[str], exp10_dir: Path) -> None:
    stems = stems[:TOP_N]
    out_dir = OUT_ROOT / name
    img_dir = out_dir / "images"
    collage_dir = out_dir / "collages"
    img_dir.mkdir(parents=True, exist_ok=True)
    collage_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    for stem in stems:
        src_img = first_match(TEST_IMAGES, stem)
        exp8_img = first_match(EXP8_PRED_DIR, stem)
        exp10_img = first_match(exp10_dir, stem)
        if src_img is None or exp8_img is None or exp10_img is None:
            report_lines.append(f"[WARN] missing files for {stem}\n")
            continue

        safe_copy(src_img, img_dir / src_img.name)
        out = collage_dir / f"{stem}_compare.jpg"
        build_collage(exp8_img, exp10_img, out, stem)
        report_lines.append(f"{stem}\n  source: {src_img.name}\n  collage: {out.name}\n")

    (out_dir / "selected_cases.txt").write_text("".join(report_lines), encoding="utf-8")


def main() -> None:
    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Missing test images folder: {TEST_IMAGES}")
    if not EXP8_PRED_DIR.exists():
        raise FileNotFoundError(f"Missing exp8 prediction folder: {EXP8_PRED_DIR}")

    exp10_dir = find_exp10_pred_dir()

    fp_stems = read_stems(FP_TXT)
    prec_stems = read_stems(PREC_TXT)
    if not fp_stems:
        raise FileNotFoundError(f"Missing or empty candidate file: {FP_TXT}")
    if not prec_stems:
        raise FileNotFoundError(f"Missing or empty candidate file: {PREC_TXT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    process_case_list("fp_cases", fp_stems, exp10_dir)
    process_case_list("precision_cases", prec_stems, exp10_dir)

    print(f"Done. Output root: {OUT_ROOT}")
    print(f"False-positive cases: {OUT_ROOT / 'fp_cases'}")
    print(f"Precision cases: {OUT_ROOT / 'precision_cases'}")


if __name__ == "__main__":
    main()
