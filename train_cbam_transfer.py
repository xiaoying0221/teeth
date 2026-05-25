#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp6: YOLOv8s + CBAM + exp4权重精确迁移
通过层编号映射表，把 exp4 的权重精确迁移到 CBAM 模型
"""

from pathlib import Path
from copy import deepcopy
import torch
from ultralytics import YOLO

# ============================================================
# 路径配置
# ============================================================
DATA_YAML   = Path(r"d:\文档\毕业设计\teeth\dataset_yolo\data.yaml")
CBAM_YAML   = Path(r"d:\文档\毕业设计\teeth\yolov8s-cbam.yaml")
EXP4_WEIGHT = Path(r"d:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\weights\best.pt")
OUT_DIR     = Path(r"d:\文档\毕业设计\teeth\runs")
NAME        = "exp6_yolov8s_cbam_transfer"
EPOCHS      = 200
BATCH       = 16
DEVICE      = "0"
WORKERS     = 4

# ============================================================
# exp4层编号 -> CBAM层编号 映射
# CBAM在 layer5/8/11 插入，导致后续层号偏移
# ============================================================
LAYER_MAP = {
    '0':  '0',   # Conv P1
    '1':  '1',   # Conv P2
    '2':  '2',   # C2f
    '3':  '3',   # Conv P3
    '4':  '4',   # C2f P3  -> CBAM layer5 是新增的
    '5':  '6',   # Conv P4  (exp4.5 -> cbam.6)
    '6':  '7',   # C2f P4  -> CBAM layer8 是新增的
    '7':  '9',   # Conv P5  (exp4.7 -> cbam.9)
    '8':  '10',  # C2f P5  -> CBAM layer11 是新增的
    '9':  '12',  # SPPF     (exp4.9 -> cbam.12)
    # Head
    '12': '15',  # C2f
    '15': '18',  # C2f P3
    '16': '19',  # Conv
    '18': '21',  # C2f P4
    '19': '22',  # Conv
    '21': '24',  # C2f P5
    '22': '25',  # Detect
}


def remap_state_dict(exp4_sd, cbam_sd):
    """按映射表重命名 exp4 的 key，然后加载到 cbam 模型"""
    remapped = {}
    for k, v in exp4_sd.items():
        parts = k.split('.')  # ['model', '5', 'conv', 'weight']
        if parts[0] == 'model' and parts[1] in LAYER_MAP:
            new_layer = LAYER_MAP[parts[1]]
            new_key = '.'.join(['model', new_layer] + parts[2:])
            if new_key in cbam_sd and cbam_sd[new_key].shape == v.shape:
                remapped[new_key] = v
    return remapped


def transfer_weights():
    print("[1/3] 初始化 YOLOv8s-CBAM 模型结构...")
    cbam_model = YOLO(str(CBAM_YAML)).model
    cbam_sd = cbam_model.state_dict()

    print("[2/3] 加载 exp4 权重并按映射表迁移...")
    ckpt = torch.load(str(EXP4_WEIGHT), map_location='cpu', weights_only=False)
    exp4_sd = ckpt['model'].float().state_dict()

    remapped = remap_state_dict(exp4_sd, cbam_sd)
    cbam_model.load_state_dict(remapped, strict=False)

    total  = len(cbam_sd)
    copied = len(remapped)
    cbam_only = total - copied
    print(f"[3/3] 迁移完成:")
    print(f"  匹配层数: {copied}/{total} ({copied/total*100:.1f}%)")
    print(f"  CBAM新增层（随机初始化）: {cbam_only} 个参数张量")

    # 保存为临时 checkpoint
    save_path = OUT_DIR / f"{NAME}_init.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': deepcopy(cbam_model).half(), 'epoch': -1}, str(save_path))
    print(f"  初始化权重已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("exp6: YOLOv8s + CBAM + exp4权重精确迁移")
    print("=" * 60)

    init_pt = transfer_weights()

    print("\n开始训练...")
    model = YOLO(str(init_pt))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUT_DIR),
        name=NAME,
        exist_ok=True,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\n" + "=" * 60)
    print("训练完成！结果保存在:", OUT_DIR / NAME)
    print("=" * 60)

    print("\n[验证]")
    metrics = model.val()
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n[测试集推理]")
    model.predict(
        source=str(Path(r"d:\文档\毕业设计\teeth\dataset_yolo\images\test")),
        save=True,
        save_txt=True,
        conf=0.25,
        project=str(OUT_DIR),
        name=NAME + "_predict",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
