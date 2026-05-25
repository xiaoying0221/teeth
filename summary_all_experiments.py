# -*- coding: utf-8 -*-
"""
汇总所有实验训练结果，并对已跑过 test 的实验统一汇总测试集结果

训练结果：读取 runs/*/results.csv
测试结果：对已完成测试的实验 best.pt 重新在 test 集评估，统一打印结果
"""

import csv
import math
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"d:\文档\毕业设计\teeth")
RUNS = BASE / "runs"
DATA_YAML = BASE / "dataset_yolo" / "data.yaml"

TRAIN_EXPS = [
    ("exp1", "YOLOv8n 50ep", RUNS / "exp1_yolov8n" / "results.csv"),
    ("exp2", "YOLOv8n 100ep", RUNS / "exp2_yolov8n_100ep" / "results.csv"),
    ("exp3", "YOLOv8s 100ep", RUNS / "exp3_yolov8s_100ep" / "results.csv"),
    ("exp4", "YOLOv8s 250ep", RUNS / "exp4_yolov8s_250ep" / "results.csv"),
    ("exp5", "YOLOv8s+CBAM(all)", RUNS / "exp5_yolov8s_cbam_250ep" / "results.csv"),
    ("exp6", "YOLOv8s+CBAM transfer", RUNS / "exp6_yolov8s_cbam_transfer" / "results.csv"),
    ("exp7", "YOLOv8s finetune 200ep", RUNS / "exp7_yolov8s_200ep_finetune" / "results.csv"),
    ("exp8", "YOLOv8s scratch 250ep", RUNS / "exp8_yolov8s_250ep_scratch" / "results.csv"),
    ("exp9", "YOLOv8s+ECA 250ep", RUNS / "exp9_yolov8s_eca_250ep" / "results.csv"),
    ("exp10", "YOLOv8s+CBAM(P4)", RUNS / "exp10_yolov8s_cbam_p4_250ep" / "results.csv"),
    ("exp11", "YOLOv8s+CBAM(P4+P5)", RUNS / "exp11_yolov8s_cbam_p4p5_250ep" / "results.csv"),
    ("exp12", "YOLOv8s+CBAM(Neck)", RUNS / "exp12_yolov8s_cbam_neck_250ep" / "results.csv"),
]

TEST_EXPS = [
    ("exp3", "YOLOv8s 100ep", RUNS / "exp3_yolov8s_100ep" / "weights" / "best.pt", "exp3_test_eval_resummary"),
    ("exp5", "YOLOv8s+CBAM(all)", RUNS / "exp5_yolov8s_cbam_250ep" / "weights" / "best.pt", "exp5_test_eval_resummary"),
    ("exp6", "YOLOv8s+CBAM transfer", RUNS / "exp6_yolov8s_cbam_transfer" / "weights" / "best.pt", "exp6_test_eval_resummary"),
    ("exp7", "YOLOv8s finetune 200ep", RUNS / "exp7_yolov8s_200ep_finetune" / "weights" / "best.pt", "exp7_test_eval_resummary"),
    ("exp8", "YOLOv8s scratch 250ep", RUNS / "exp8_yolov8s_250ep_scratch" / "weights" / "best.pt", "exp8_test_eval_resummary"),
    ("exp10", "YOLOv8s+CBAM(P4)", RUNS / "exp10_yolov8s_cbam_p4_250ep" / "weights" / "best.pt", "exp10_test_eval_resummary"),
    ("exp11", "YOLOv8s+CBAM(P4+P5)", RUNS / "exp11_yolov8s_cbam_p4p5_250ep" / "weights" / "best.pt", "exp11_test_eval_resummary"),
    ("exp12", "YOLOv8s+CBAM(Neck)", RUNS / "exp12_yolov8s_cbam_neck_250ep" / "weights" / "best.pt", "exp12_test_eval_resummary"),
]


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize_train(rows):
    epochs = [int(r['epoch']) for r in rows]
    m50 = [float(r['metrics/mAP50(B)']) for r in rows]
    m95 = [float(r['metrics/mAP50-95(B)']) for r in rows]
    p = [float(r['metrics/precision(B)']) for r in rows]
    r = [float(r['metrics/recall(B)']) for r in rows]
    last = rows[-1]
    last50 = m50[-50:] if len(m50) >= 50 else m50
    mean50 = sum(last50) / len(last50)
    std50 = math.sqrt(sum((x - mean50) ** 2 for x in last50) / len(last50))
    return {
        'epochs': epochs[-1],
        'best_m50': max(m50),
        'best_m50_ep': epochs[m50.index(max(m50))],
        'best_m95': max(m95),
        'best_m95_ep': epochs[m95.index(max(m95))],
        'last_p': float(last['metrics/precision(B)']),
        'last_r': float(last['metrics/recall(B)']),
        'last_m50': float(last['metrics/mAP50(B)']),
        'last_m95': float(last['metrics/mAP50-95(B)']),
        'mean50': mean50,
        'std50': std50,
    }


def print_train_summary():
    print('=' * 110)
    print('所有实验训练结果汇总')
    print('=' * 110)
    print('{:6s} {:26s} {:>7s} {:>10s} {:>8s} {:>12s} {:>8s} {:>10s} {:>10s}'.format(
        '实验', '模型', '轮次', 'best50', 'bestEp', 'best50-95', '末轮P', '末轮R', '收敛std'))
    print('-' * 110)

    for exp_id, name, path in TRAIN_EXPS:
        if not path.exists():
            print('{:6s} {:26s} {}'.format(exp_id, name, '结果不存在'))
            continue
        s = summarize_train(read_csv(path))
        print('{:6s} {:26s} {:>7d} {:>10.4f} {:>8d} {:>12.4f} {:>8.4f} {:>10.4f} {:>10.5f}'.format(
            exp_id, name, s['epochs'], s['best_m50'], s['best_m50_ep'], s['best_m95'], s['last_p'], s['last_r'], s['std50']))
    print('=' * 110)


def run_test_summary():
    print('\n' + '=' * 110)
    print('已跑过测试集 / 需要测试集汇总的实验结果')
    print('=' * 110)
    print('{:6s} {:26s} {:>10s} {:>10s} {:>10s} {:>12s}'.format(
        '实验', '模型', 'P', 'R', 'mAP50', 'mAP50-95'))
    print('-' * 110)

    for exp_id, name, best_pt, eval_name in TEST_EXPS:
        if not best_pt.exists():
            print('{:6s} {:26s} {}'.format(exp_id, name, '权重不存在'))
            continue

        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(DATA_YAML),
            split='test',
            imgsz=640,
            batch=16,
            device='0',
            project=str(RUNS),
            name=eval_name,
            exist_ok=True,
            verbose=False,
        )
        print('{:6s} {:26s} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}'.format(
            exp_id, name, metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map))
    print('=' * 110)


def main():
    print_train_summary()
    run_test_summary()


if __name__ == '__main__':
    main()
