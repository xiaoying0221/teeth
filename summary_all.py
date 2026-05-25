# -*- coding: utf-8 -*-
import csv, math, os

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def summarize(rows):
    epochs = [int(r['epoch']) for r in rows]
    m50  = [float(r['metrics/mAP50(B)']) for r in rows]
    m95  = [float(r['metrics/mAP50-95(B)']) for r in rows]
    p    = [float(r['metrics/precision(B)']) for r in rows]
    r    = [float(r['metrics/recall(B)']) for r in rows]
    return {
        'run_ep':      epochs[-1],
        'best_m50':    max(m50),
        'best_m50_ep': epochs[m50.index(max(m50))],
        'best_m95':    max(m95),
        'last_p':      float(rows[-1]['metrics/precision(B)']),
        'last_r':      float(rows[-1]['metrics/recall(B)']),
    }

exps = [
    ('exp1', 'YOLOv8n',         '从零',            50,
     r'd:\文档\毕业设计\teeth\runs\exp1_yolov8n\results.csv'),
    ('exp2', 'YOLOv8n',         '从零',            100,
     r'd:\文档\毕业设计\teeth\runs\exp2_yolov8n_100ep\results.csv'),
    ('exp3', 'YOLOv8s',         '从零',            100,
     r'd:\文档\毕业设计\teeth\runs\exp3_yolov8s_100ep\results.csv'),
    ('exp4', 'YOLOv8s',         '从exp3续训',      250,
     r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv'),
    ('exp5', 'YOLOv8s+CBAM',    '从零',            250,
     r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv'),
    ('exp6', 'YOLOv8s+CBAM',    '从exp4迁移',      200,
     r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv'),
    ('exp7', 'YOLOv8s',         '从exp4续训',      200,
     r'd:\文档\毕业设计\teeth\runs\exp7_yolov8s_200ep_finetune\results.csv'),
]

print('='*90)
print('{:6s} {:18s} {:14s} {:8s} {:8s} {:10s} {:12s} {:8s} {:8s}'.format(
    '实验','模型','起点','设定轮','实跑轮','最佳mAP50','最佳mAP50-95','最佳Ep','Precision'))
print('-'*90)
for exp_id, model, start, setting_ep, path in exps:
    try:
        s = summarize(read_csv(path))
        print('{:6s} {:18s} {:14s} {:8d} {:8d} {:10.4f} {:12.4f} {:8d} {:8.4f}'.format(
            exp_id, model, start, setting_ep, s['run_ep'],
            s['best_m50'], s['best_m95'], s['best_m50_ep'], s['last_p']))
    except Exception as e:
        print('{:6s} {:18s} {:14s} {:8d}  读取失败: {}'.format(exp_id, model, start, setting_ep, e))
print('='*90)
