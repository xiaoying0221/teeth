# -*- coding: utf-8 -*-
import csv
import math

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

exp4 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv')
exp5 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv')
exp6 = read_csv(r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv')

def summarize(rows):
    epochs = [int(r['epoch']) for r in rows]
    m50  = [float(r['metrics/mAP50(B)']) for r in rows]
    m95  = [float(r['metrics/mAP50-95(B)']) for r in rows]
    p    = [float(r['metrics/precision(B)']) for r in rows]
    r    = [float(r['metrics/recall(B)']) for r in rows]
    last = rows[-1]
    last50 = m50[-50:]
    mean50 = sum(last50)/len(last50)
    std50  = math.sqrt(sum((x-mean50)**2 for x in last50)/len(last50))
    return {
        'total_ep':   epochs[-1],
        'best_m50':   max(m50),
        'best_m50_ep':epochs[m50.index(max(m50))],
        'best_m95':   max(m95),
        'best_m95_ep':epochs[m95.index(max(m95))],
        'last_m50':   float(last['metrics/mAP50(B)']),
        'last_m95':   float(last['metrics/mAP50-95(B)']),
        'last_p':     float(last['metrics/precision(B)']),
        'last_r':     float(last['metrics/recall(B)']),
        'last_vbox':  float(last['val/box_loss']),
        'last_vcls':  float(last['val/cls_loss']),
        'std50':      std50,
        'mean50':     mean50,
    }

s4, s5, s6 = summarize(exp4), summarize(exp5), summarize(exp6)

print('='*60)
print('  exp6 YOLOv8s+CBAM+迁移  最新结果')
print('='*60)
print('总轮次: {}'.format(s6['total_ep']))
print()
print('[最终轮次指标]')
print('  Precision  : {:.4f}'.format(s6['last_p']))
print('  Recall     : {:.4f}'.format(s6['last_r']))
print('  mAP@50     : {:.4f}'.format(s6['last_m50']))
print('  mAP@50-95  : {:.4f}'.format(s6['last_m95']))
print('  val/box_loss: {:.5f}'.format(s6['last_vbox']))
print('  val/cls_loss: {:.5f}'.format(s6['last_vcls']))
print()
print('[最佳指标]')
print('  最高 mAP@50    : {:.4f}  @ Epoch {}'.format(s6['best_m50'], s6['best_m50_ep']))
print('  最高 mAP@50-95 : {:.4f}  @ Epoch {}'.format(s6['best_m95'], s6['best_m95_ep']))
print()
print('[收敛分析] 后50轮均值={:.4f}  标准差={:.5f}'.format(s6['mean50'], s6['std50']))
print('  收敛状态: {}'.format('已收敛(std<0.002)' if s6['std50'] < 0.002 else '基本收敛(std<0.003)' if s6['std50'] < 0.003 else '仍在震荡'))
print()
print('='*60)
print('  全部实验最终对比')
print('='*60)
all_exps = [
    ('exp1 YOLOv8n 50ep',          0.7371, 0.3843, '-',  48),
    ('exp2 YOLOv8n 100ep',         0.7860, 0.4903, '-',  90),
    ('exp3 YOLOv8s 100ep',         0.8037, 0.5313, '-',  98),
    ('exp4 YOLOv8s 250ep',         s4['best_m50'], s4['best_m95'], 'No',  s4['best_m50_ep']),
    ('exp5 YOLOv8s+CBAM 250ep',    s5['best_m50'], s5['best_m95'], 'CBAM', s5['best_m50_ep']),
    ('exp6 YOLOv8s+CBAM+迁移',     s6['best_m50'], s6['best_m95'], 'CBAM+transfer', s6['best_m50_ep']),
]
all_exps.sort(key=lambda x: x[1], reverse=True)
print('{:3s}  {:<28s} {:>10s}  {:>12s}  {:>8s}'.format('排名','实验','mAP@50','mAP@50-95','最佳Ep'))
print('-'*65)
for i,(name,m50,m95,attn,ep) in enumerate(all_exps, 1):
    print('{}    {:<28s} {:.4f}        {:.4f}      Ep {}'.format(i, name, m50, m95, ep))
print()
print('vs exp4 (best):')
print('  exp6 mAP50:    {:.4f} -> {:.4f}  ({:+.4f})'.format(s4['best_m50'], s6['best_m50'], s6['best_m50']-s4['best_m50']))
print('  exp6 mAP50-95: {:.4f} -> {:.4f}  ({:+.4f})'.format(s4['best_m95'], s6['best_m95'], s6['best_m95']-s4['best_m95']))
print('='*60)
