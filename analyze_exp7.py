# -*- coding: utf-8 -*-
import csv
import math

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def summarize(rows):
    epochs = [int(r['epoch']) for r in rows]
    m50  = [float(r['metrics/mAP50(B)']) for r in rows]
    m95  = [float(r['metrics/mAP50-95(B)']) for r in rows]
    p    = [float(r['metrics/precision(B)']) for r in rows]
    r    = [float(r['metrics/recall(B)']) for r in rows]
    last = rows[-1]
    last50 = m50[-50:] if len(m50) >= 50 else m50
    mean50 = sum(last50)/len(last50)
    std50  = math.sqrt(sum((x-mean50)**2 for x in last50)/len(last50))
    return {
        'total_ep':    epochs[-1],
        'best_m50':    max(m50),
        'best_m50_ep': epochs[m50.index(max(m50))],
        'best_m95':    max(m95),
        'best_m95_ep': epochs[m95.index(max(m95))],
        'last_m50':    float(last['metrics/mAP50(B)']),
        'last_m95':    float(last['metrics/mAP50-95(B)']),
        'last_p':      float(last['metrics/precision(B)']),
        'last_r':      float(last['metrics/recall(B)']),
        'std50':       std50,
        'mean50':      mean50,
    }

s7 = summarize(read_csv(r'd:\文档\毕业设计\teeth\runs\exp7_yolov8s_200ep_finetune\results.csv'))
s6 = summarize(read_csv(r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv'))

print('='*60)
print('exp7  YOLOv8s（无CBAM）200轮 结果')
print('='*60)
print('总轮次:', s7['total_ep'])
print('最终: P={:.4f}  R={:.4f}  mAP50={:.4f}  mAP50-95={:.4f}'.format(
    s7['last_p'], s7['last_r'], s7['last_m50'], s7['last_m95']))
print('最佳 mAP50={:.4f} @Epoch {}  |  最佳 mAP50-95={:.4f} @Epoch {}'.format(
    s7['best_m50'], s7['best_m50_ep'], s7['best_m95'], s7['best_m95_ep']))
print('后50轮均值={:.4f}  标准差={:.5f}  ({})'.format(
    s7['mean50'], s7['std50'],
    '已收敛' if s7['std50'] < 0.002 else '基本收敛' if s7['std50'] < 0.003 else '仍震荡'))

print()
print('='*60)
print('控制变量对比：YOLOv8s vs YOLOv8s+CBAM（同样起点，同样200轮）')
print('='*60)
print('{:20s} {:>12s} {:>14s} {:>10s} {:>10s}'.format('模型','最佳mAP50','最佳mAP50-95','末轮P','末轮R'))
print('-'*68)
print('{:20s} {:>12.4f} {:>14.4f} {:>10.4f} {:>10.4f}'.format(
    'exp7 YOLOv8s', s7['best_m50'], s7['best_m95'], s7['last_p'], s7['last_r']))
print('{:20s} {:>12.4f} {:>14.4f} {:>10.4f} {:>10.4f}'.format(
    'exp6 YOLOv8s+CBAM', s6['best_m50'], s6['best_m95'], s6['last_p'], s6['last_r']))
print()
diff50  = s6['best_m50']  - s7['best_m50']
diff95  = s6['best_m95']  - s7['best_m95']
print('CBAM 带来的提升:')
print('  mAP50:    {:+.4f} ({:+.2f}%)'.format(diff50,  diff50*100))
print('  mAP50-95: {:+.4f} ({:+.2f}%)'.format(diff95, diff95*100))
print('='*60)
