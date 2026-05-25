# -*- coding: utf-8 -*-
import csv
import math


def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows

exp4 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv')
exp5 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv')
exp6 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv')


def summarize(rows):
    epochs = [int(r['epoch']) for r in rows]
    m50 = [float(r['metrics/mAP50(B)']) for r in rows]
    m95 = [float(r['metrics/mAP50-95(B)']) for r in rows]
    p = [float(r['metrics/precision(B)']) for r in rows]
    r = [float(r['metrics/recall(B)']) for r in rows]
    last = rows[-1]
    return {
        'epochs': epochs[-1],
        'last_m50': float(last['metrics/mAP50(B)']),
        'last_m95': float(last['metrics/mAP50-95(B)']),
        'last_p': float(last['metrics/precision(B)']),
        'last_r': float(last['metrics/recall(B)']),
        'best_m50': max(m50),
        'best_m50_ep': epochs[m50.index(max(m50))],
        'best_m95': max(m95),
        'best_m95_ep': epochs[m95.index(max(m95))],
    }

s4, s5, s6 = summarize(exp4), summarize(exp5), summarize(exp6)

print('='*60)
print('exp6 结果')
print('='*60)
print('总轮次:', s6['epochs'])
print('最终: P={:.4f}, R={:.4f}, mAP50={:.4f}, mAP50-95={:.4f}'.format(s6['last_p'], s6['last_r'], s6['last_m50'], s6['last_m95']))
print('最佳 mAP50={:.4f} @Epoch {}'.format(s6['best_m50'], s6['best_m50_ep']))
print('最佳 mAP50-95={:.4f} @Epoch {}'.format(s6['best_m95'], s6['best_m95_ep']))

print('\n与 exp4 对比（best）:')
print('mAP50:    {:.4f} -> {:.4f}  ({:+.4f})'.format(s4['best_m50'], s6['best_m50'], s6['best_m50']-s4['best_m50']))
print('mAP50-95: {:.4f} -> {:.4f}  ({:+.4f})'.format(s4['best_m95'], s6['best_m95'], s6['best_m95']-s4['best_m95']))

print('\n与 exp5 对比（best）:')
print('mAP50:    {:.4f} -> {:.4f}  ({:+.4f})'.format(s5['best_m50'], s6['best_m50'], s6['best_m50']-s5['best_m50']))
print('mAP50-95: {:.4f} -> {:.4f}  ({:+.4f})'.format(s5['best_m95'], s6['best_m95'], s6['best_m95']-s5['best_m95']))

print('\n当前最佳实验（按 best mAP50）:')
all_exps = [('exp4', s4['best_m50']), ('exp5', s5['best_m50']), ('exp6', s6['best_m50'])]
all_exps.sort(key=lambda x: x[1], reverse=True)
for i, (n, v) in enumerate(all_exps, 1):
    print('{}. {}  {:.4f}'.format(i, n, v))
print('='*60)
