# -*- coding: utf-8 -*-
import csv
import math

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

rows5 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv')
rows4 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv')

def metrics(rows):
    map50   = [float(r['metrics/mAP50(B)'])    for r in rows]
    map5095 = [float(r['metrics/mAP50-95(B)']) for r in rows]
    prec    = [float(r['metrics/precision(B)']) for r in rows]
    rec     = [float(r['metrics/recall(B)'])    for r in rows]
    epochs  = [int(r['epoch'])                  for r in rows]
    return map50, map5095, prec, rec, epochs

m50_5, m5095_5, p5, r5, ep5 = metrics(rows5)
m50_4, m5095_4, p4, r4, ep4 = metrics(rows4)

last5 = rows5[-1]
last4 = rows4[-1]

print('=' * 60)
print('  exp5  YOLOv8s + CBAM  训练结果')
print('=' * 60)
print('总轮次 :', ep5[-1])
print()
print('[最终轮次指标]')
print('  Precision  :', round(float(last5['metrics/precision(B)']), 4))
print('  Recall     :', round(float(last5['metrics/recall(B)']), 4))
print('  mAP@50     :', round(float(last5['metrics/mAP50(B)']), 4))
print('  mAP@50-95  :', round(float(last5['metrics/mAP50-95(B)']), 4))
print('  val/box_loss:', round(float(last5['val/box_loss']), 5))
print('  val/cls_loss:', round(float(last5['val/cls_loss']), 5))
print('  val/dfl_loss:', round(float(last5['val/dfl_loss']), 5))
print()
print('[最佳指标]')
print('  最高 mAP@50    :', round(max(m50_5), 4), ' @ Epoch', ep5[m50_5.index(max(m50_5))])
print('  最高 mAP@50-95 :', round(max(m5095_5), 4), ' @ Epoch', ep5[m5095_5.index(max(m5095_5))])
print('  最高 Precision :', round(max(p5), 4), ' @ Epoch', ep5[p5.index(max(p5))])
print('  最高 Recall    :', round(max(r5), 4), ' @ Epoch', ep5[r5.index(max(r5))])

# 收敛判断
last50 = m50_5[-50:]
mean50 = sum(last50) / len(last50)
std50  = math.sqrt(sum((x-mean50)**2 for x in last50) / len(last50))
print()
print('[收敛分析] 后50轮 mAP@50 均值:', round(mean50,4), ' 标准差:', round(std50,5))
print('  收敛判断:', '已收敛(std<0.003)' if std50 < 0.003 else '未完全收敛')

print()
print('=' * 60)
print('  与 exp4 (YOLOv8s 250ep, 无CBAM) 对比')
print('=' * 60)
rows = [
    ('最佳 mAP@50',    max(m50_4),   max(m50_5)),
    ('最佳 mAP@50-95', max(m5095_4), max(m5095_5)),
    ('末轮 mAP@50',    float(last4['metrics/mAP50(B)']),    float(last5['metrics/mAP50(B)'])),
    ('末轮 mAP@50-95', float(last4['metrics/mAP50-95(B)']), float(last5['metrics/mAP50-95(B)'])),
    ('末轮 Precision',  float(last4['metrics/precision(B)']), float(last5['metrics/precision(B)'])),
    ('末轮 Recall',     float(last4['metrics/recall(B)']),    float(last5['metrics/recall(B)'])),
]
print(f"  {'指标':<16} {'exp4 无CBAM':>12} {'exp5 +CBAM':>12} {'提升':>10}")
print('  ' + '-'*52)
for label, v4, v5 in rows:
    diff = v5 - v4
    sign = '+' if diff >= 0 else ''
    print(f"  {label:<16} {v4:>12.4f} {v5:>12.4f} {sign+str(round(diff,4)):>10}")

print()
print('[所有实验汇总 - 最佳 mAP@50]')
exps = [
    ('exp1 YOLOv8n 50ep',    r'd:\文档\毕业设计\teeth\runs\exp1_yolov8n\results.csv'),
    ('exp2 YOLOv8n 100ep',   r'd:\文档\毕业设计\teeth\runs\exp2_yolov8n_100ep\results.csv'),
    ('exp3 YOLOv8s 100ep',   r'd:\文档\毕业设计\teeth\runs\exp3_yolov8s_100ep\results.csv'),
    ('exp4 YOLOv8s 250ep',   r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv'),
    ('exp5 YOLOv8s+CBAM',    r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv'),
]
for name, path in exps:
    try:
        rr = read_csv(path)
        vals = [float(r['metrics/mAP50(B)']) for r in rr]
        ep   = [int(r['epoch']) for r in rr]
        best = max(vals)
        best_ep = ep[vals.index(best)]
        print(f"  {name:<25} best mAP@50={best:.4f}  @ Epoch {best_ep}")
    except Exception as e:
        print(f"  {name:<25} ERROR: {e}")
print('=' * 60)
