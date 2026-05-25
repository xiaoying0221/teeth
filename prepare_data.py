#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗与准备脚本
功能：
1. 从 dataset4 提取已有 YOLO 标注数据，统一类别为 abnormal(0)
2. 从 dataset3 转换 Pascal VOC XML 为 YOLO txt
3. 合并所有数据，划分 train/val/test
4. 生成 data.yaml
5. 验证标注（越界检查、坏图检查）
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import glob

random.seed(42)

# ============================================================
# 路径配置
# ============================================================
BASE = Path(r"d:\文档\毕业设计\teeth")

# dataset4: 已有 YOLO 标注（多类，需统一）
DS4_IMAGES_TRAIN = BASE / "tooth/dataset4/archive/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/train"
DS4_IMAGES_VAL   = BASE / "tooth/dataset4/archive/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/val"
DS4_LABELS_TRAIN = BASE / "tooth/dataset4/archive/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/train"
DS4_LABELS_VAL   = BASE / "tooth/dataset4/archive/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/val"

# dataset3: X光片 + Pascal VOC XML
DS3_IMAGES = BASE / "tooth/dataset3/Dental Cavity Dataset/Dataset/x-ray/images"
DS3_XMLS   = BASE / "tooth/dataset3/Dental Cavity Dataset/Dataset/x-ray/xmls"

# 输出目录
OUT = BASE / "dataset_yolo"
OUT_IMAGES_TRAIN = OUT / "images/train"
OUT_IMAGES_VAL   = OUT / "images/val"
OUT_IMAGES_TEST  = OUT / "images/test"
OUT_LABELS_TRAIN = OUT / "labels/train"
OUT_LABELS_VAL   = OUT / "labels/val"
OUT_LABELS_TEST  = OUT / "labels/test"

# ============================================================
# 工具函数
# ============================================================

def ensure_dirs():
    for d in [OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_IMAGES_TEST,
              OUT_LABELS_TRAIN, OUT_LABELS_VAL, OUT_LABELS_TEST]:
        d.mkdir(parents=True, exist_ok=True)


def is_valid_image(path):
    """检查图像是否能正常打开"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def fix_yolo_label(label_path):
    """
    读取 YOLO label，把所有类别 ID 改为 0，修复越界坐标。
    返回修复后的行列表，如果文件为空或无效返回 None。
    """
    lines = Path(label_path).read_text(encoding='utf-8').strip().splitlines()
    fixed = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        # 类别统一为 0
        cls = 0
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # 修复越界
        cx = clamp(cx)
        cy = clamp(cy)
        w  = clamp(w)
        h  = clamp(h)
        # 确保框不超出图像边界
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        x1 = clamp(x1)
        y1 = clamp(y1)
        x2 = clamp(x2)
        y2 = clamp(y2)
        if x2 <= x1 or y2 <= y1:
            continue  # 无效框，跳过
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        fixed.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return fixed if fixed else None


def xml_to_yolo(xml_path, img_path):
    """
    Pascal VOC XML -> YOLO txt 格式
    返回行列表，如果无有效框返回 None。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [XML解析失败] {xml_path}: {e}")
        return None

    size = root.find('size')
    if size is None:
        # 从图像读取尺寸
        try:
            with Image.open(img_path) as img:
                W, H = img.size
        except Exception:
            return None
    else:
        W = int(size.find('width').text)
        H = int(size.find('height').text)

    if W <= 0 or H <= 0:
        return None

    lines = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # 归一化
        xmin = clamp(xmin / W)
        ymin = clamp(ymin / H)
        xmax = clamp(xmax / W)
        ymax = clamp(ymax / H)

        if xmax <= xmin or ymax <= ymin:
            continue

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w  = xmax - xmin
        h  = ymax - ymin
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines if lines else None


# ============================================================
# 步骤 1：收集 dataset4 数据
# ============================================================

def collect_dataset4():
    """收集 dataset4 中有配对标注的图像，统一类别为 0"""
    print("\n[Step 1] 收集 dataset4 (YOLO标注)...")
    collected = []  # list of (img_path, label_lines)
    bad = 0
    no_label = 0

    for img_dir, lbl_dir in [(DS4_IMAGES_TRAIN, DS4_LABELS_TRAIN),
                              (DS4_IMAGES_VAL,   DS4_LABELS_VAL)]:
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        img_files = []
        for ext in img_extensions:
            img_files.extend(img_dir.glob(ext))

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                no_label += 1
                continue
            if not is_valid_image(img_path):
                print(f"  [坏图] {img_path.name}")
                bad += 1
                continue
            fixed = fix_yolo_label(lbl_path)
            if fixed is None:
                no_label += 1
                continue
            collected.append((img_path, fixed))

    print(f"  有效样本: {len(collected)}  |  无标注: {no_label}  |  坏图: {bad}")
    return collected


# ============================================================
# 步骤 2：收集 dataset3 数据（XML -> YOLO）
# ============================================================

def collect_dataset3():
    """收集 dataset3 X光片，XML转YOLO格式"""
    print("\n[Step 2] 收集 dataset3 (XML->YOLO)...")
    collected = []
    bad = 0
    no_label = 0
    no_img = 0

    xml_files = list(DS3_XMLS.glob('*.xml'))
    for xml_path in xml_files:
        stem = xml_path.stem
        # 找对应图像
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            candidate = DS3_IMAGES / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            no_img += 1
            continue
        if not is_valid_image(img_path):
            print(f"  [坏图] {img_path.name}")
            bad += 1
            continue
        lines = xml_to_yolo(xml_path, img_path)
        if lines is None:
            no_label += 1
            continue
        collected.append((img_path, lines))

    print(f"  有效样本: {len(collected)}  |  无对应图像: {no_img}  |  无有效框: {no_label}  |  坏图: {bad}")
    return collected


# ============================================================
# 步骤 3：合并并划分 train/val/test
# ============================================================

def split_and_save(all_samples):
    """随机打乱后按 7:2:1 划分并保存"""
    print(f"\n[Step 3] 合并 {len(all_samples)} 个样本，划分 train/val/test...")
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.2)
    # 剩余给 test

    splits = {
        'train': all_samples[:n_train],
        'val':   all_samples[n_train:n_train + n_val],
        'test':  all_samples[n_train + n_val:],
    }

    img_dirs = {
        'train': OUT_IMAGES_TRAIN,
        'val':   OUT_IMAGES_VAL,
        'test':  OUT_IMAGES_TEST,
    }
    lbl_dirs = {
        'train': OUT_LABELS_TRAIN,
        'val':   OUT_LABELS_VAL,
        'test':  OUT_LABELS_TEST,
    }

    counters = {}
    for split, samples in splits.items():
        for i, (img_path, label_lines) in enumerate(samples):
            # 避免文件名冲突，用 split_序号_原文件名
            new_stem = f"{split}_{i:04d}_{img_path.stem}"
            # 保持原始扩展名（统一转为小写 .jpg）
            dst_img = img_dirs[split] / (new_stem + '.jpg')
            dst_lbl = lbl_dirs[split] / (new_stem + '.txt')

            # 复制图像（转换为 RGB jpg）
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img.save(dst_img, 'JPEG', quality=95)
            except Exception as e:
                print(f"  [保存失败] {img_path.name}: {e}")
                continue

            # 写标注
            dst_lbl.write_text('\n'.join(label_lines), encoding='utf-8')

        counters[split] = len(samples)
        print(f"  {split}: {len(samples)} 张")

    return counters


# ============================================================
# 步骤 4：生成 data.yaml
# ============================================================

def write_yaml():
    yaml_content = f"""# 口腔影像异常区域检测数据集
path: {OUT.as_posix()}  # 数据集根目录
train: images/train
val: images/val
test: images/test

nc: 1  # 类别数量
names: ['abnormal']  # 类别名称
"""
    yaml_path = OUT / 'data.yaml'
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"\n[Step 4] data.yaml 已生成: {yaml_path}")


# ============================================================
# 步骤 5：验证输出
# ============================================================

def verify_output():
    print("\n[Step 5] 验证输出...")
    for split in ['train', 'val', 'test']:
        img_dir = OUT / f'images/{split}'
        lbl_dir = OUT / f'labels/{split}'
        imgs = list(img_dir.glob('*.jpg'))
        lbls = list(lbl_dir.glob('*.txt'))
        # 检查是否每张图都有对应 label
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        missing = img_stems - lbl_stems
        if missing:
            print(f"  [{split}] 警告: {len(missing)} 张图缺少 label")
        else:
            print(f"  [{split}] OK: {len(imgs)} 张图，{len(lbls)} 个标注，全部配对")


# ============================================================
# 主流程
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("口腔影像数据集准备脚本")
    print("=" * 60)

    ensure_dirs()

    samples_ds4 = collect_dataset4()
    samples_ds3 = collect_dataset3()

    all_samples = samples_ds4 + samples_ds3
    print(f"\n总样本数: {len(all_samples)}")

    if len(all_samples) == 0:
        print("错误：没有找到任何有效样本，请检查数据路径！")
        exit(1)

    split_and_save(all_samples)
    write_yaml()
    verify_output()

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print(f"输出目录: {OUT}")
    print("下一步: 运行 train.py 开始训练")
    print("=" * 60)
        