# -*- coding: utf-8 -*-
import csv
import math


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize(rows):
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
        'total_ep': epochs[-1],
        'best_m50': max(m50),
        'best_m50_ep': epochs[m50.index(max(m50))],
        'best_m95': max(m95),
        'best_m95_ep': epochs[m95.index(max(m95))],
        'last_m50': float(last['metrics/mAP50(B)']),
        'last_m95': float(last['metrics/mAP50-95(B)']),
        'last_p': float(last['metrics/precision(B)']),
        'last_r': float(last['metrics/recall(B)']),
        'mean50': mean50,
        'std50': std50,
    }


def load_or_none(path):
    try:
        return summarize(read_csv(path))
    except Exception:
        return None


exp5 = load_or_none(r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv')
exp8 = load_or_none(r'd:\文档\毕业设计\teeth\runs\exp8_yolov8s_250ep_scratch\results.csv')
exp10 = load_or_none(r'd:\文档\毕业设计\teeth\runs\exp10_yolov8s_cbam_p4_250ep\results.csv')
exp11 = load_or_none(r'd:\文档\毕业设计\teeth\runs\exp11_yolov8s_cbam_p4p5_250ep\results.csv')

print('=' * 74)
print('exp11 YOLOv8s+CBAM(P4+P5) 训练结果分析')
print('=' * 74)

if exp11 is None:
    print('exp11 结果文件还不存在，请先运行 train_cbam_p4p5.py')
    print('=' * 74)
    raise SystemExit(0)

print('总轮次: {}'.format(exp11['total_ep']))
print('最终: P={:.4f}  R={:.4f}  mAP50={:.4f}  mAP50-95={:.4f}'.format(
    exp11['last_p'], exp11['last_r'], exp11['last_m50'], exp11['last_m95']))
print('最佳 mAP50={:.4f} @Epoch {}'.format(exp11['best_m50'], exp11['best_m50_ep']))
print('最佳 mAP50-95={:.4f} @Epoch {}'.format(exp11['best_m95'], exp11['best_m95_ep']))
print('后50轮均值={:.4f}  标准差={:.5f}  ({})'.format(
    exp11['mean50'], exp11['std50'],
    '已收敛' if exp11['std50'] < 0.002 else '基本收敛' if exp11['std50'] < 0.003 else '仍震荡'))

print()
print('=' * 74)
print('公平对比：exp8 vs exp5(CBAM全加) vs exp10(CBAM-P4) vs exp11(CBAM-P4+P5)')
print('（同起点：官方预训练；同轮次：250轮）')
print('=' * 74)
print('{:24s} {:>10s} {:>12s} {:>10s} {:>10s} {:>8s}'.format(
    '模型', 'best50', 'best50-95', '末轮P', '末轮R', 'bestEp'))
print('-' * 86)

for name, s in [
    ('exp8 YOLOv8s', exp8),
    ('exp5 YOLOv8s+CBAM(all)', exp5),
    ('exp10 YOLOv8s+CBAM(P4)', exp10),
    ('exp11 YOLOv8s+CBAM(P4+P5)', exp11),
]:
    if s is None:
        print('{:24s} {}'.format(name, '结果不存在'))
    else:
        print('{:24s} {:>10.4f} {:>12.4f} {:>10.4f} {:>10.4f} {:>8d}'.format(
            name, s['best_m50'], s['best_m95'], s['last_p'], s['last_r'], s['best_m50_ep']))

if exp8 is not None:
    print()
    print('相对 exp8（无注意力）的变化:')
    print('CBAM(all)    mAP50:    {:+.4f}'.format(exp5['best_m50'] - exp8['best_m50']) if exp5 else 'CBAM(all) 结果不存在')
    print('CBAM(all)    mAP50-95: {:+.4f}'.format(exp5['best_m95'] - exp8['best_m95']) if exp5 else 'CBAM(all) 结果不存在')
    print('CBAM(P4)     mAP50:    {:+.4f}'.format(exp10['best_m50'] - exp8['best_m50']) if exp10 else 'CBAM(P4) 结果不存在')
    print('CBAM(P4)     mAP50-95: {:+.4f}'.format(exp10['best_m95'] - exp8['best_m95']) if exp10 else 'CBAM(P4) 结果不存在')
    print('CBAM(P4+P5)  mAP50:    {:+.4f}'.format(exp11['best_m50'] - exp8['best_m50']))
    print('CBAM(P4+P5)  mAP50-95: {:+.4f}'.format(exp11['best_m95'] - exp8['best_m95']))

if exp5 is not None:
    print()
    print('相对 exp5（P3+P4+P5 全加）的变化:')
    print('CBAM(P4)     mAP50:    {:+.4f}'.format(exp10['best_m50'] - exp5['best_m50']) if exp10 else 'CBAM(P4) 结果不存在')
    print('CBAM(P4)     mAP50-95: {:+.4f}'.format(exp10['best_m95'] - exp5['best_m95']) if exp10 else 'CBAM(P4) 结果不存在')
    print('CBAM(P4+P5)  mAP50:    {:+.4f}'.format(exp11['best_m50'] - exp5['best_m50']))
    print('CBAM(P4+P5)  mAP50-95: {:+.4f}'.format(exp11['best_m95'] - exp5['best_m95']))

print('=' * 74)
