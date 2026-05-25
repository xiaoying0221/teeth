# -*- coding: utf-8 -*-
import csv
import yaml

def get_info(results_path, args_path):
    with open(args_path, encoding='utf-8') as f:
        args = yaml.safe_load(f)
    rows = []
    with open(results_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    m50 = [float(r['metrics/mAP50(B)']) for r in rows]
    epochs_run = int(rows[-1]['epoch'])
    return {
        'epochs_setting': args.get('epochs', '?'),
        'epochs_run': epochs_run,
        'best_m50': max(m50),
        'best_ep': [int(r['epoch']) for r in rows][m50.index(max(m50))],
        'model': args.get('model', '?'),
    }

exps = [
    ('exp1 YOLOv8n',          r'd:\文档\毕业设计\teeth\runs\exp1_yolov8n\results.csv',                       r'd:\文档\毕业设计\teeth\runs\exp1_yolov8n\args.yaml'),
    ('exp2 YOLOv8n',          r'd:\文档\毕业设计\teeth\runs\exp2_yolov8n_100ep\results.csv',                 r'd:\文档\毕业设计\teeth\runs\exp2_yolov8n_100ep\args.yaml'),
    ('exp3 YOLOv8s',          r'd:\文档\毕业设计\teeth\runs\exp3_yolov8s_100ep\results.csv',                 r'd:\文档\毕业设计\teeth\runs\exp3_yolov8s_100ep\args.yaml'),
    ('exp4 YOLOv8s',          r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv',                 r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\args.yaml'),
    ('exp5 YOLOv8s+CBAM',     r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv',           r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\args.yaml'),
    ('exp6 YOLOv8s+CBAM',     r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv',        r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\args.yaml'),
]

print('='*70)
print('各实验实际训练情况')
print('='*70)
for name, rp, ap in exps:
    try:
        info = get_info(rp, ap)
        print('{}: 实际跑了 {} 轮，最佳mAP50={:.4f} @Epoch {}'.format(
            name, info['epochs_run'], info['best_m50'], info['best_ep']))
    except Exception as e:
        print('{}: 读取失败 ({})'.format(name, e))

print()
print('='*70)
print('控制变量视角：YOLOv8s vs YOLOv8s+CBAM')
print('='*70)

# YOLOv8s 系列：exp3(100ep) + exp4(续训到250ep) = 从零累计250轮
print('YOLOv8s（无CBAM）：')
print('  exp3: 从零训练 100 轮')
print('  exp4: 从exp3续训，累计 250 轮（相当于共训练250轮）')
print()
print('YOLOv8s+CBAM：')
print('  exp5: 从零训练 250 轮')
print('  exp6: 用exp4权重迁移初始化，再训练 200 轮')
print()
print('结论：')
print('  - exp4(YOLOv8s, 250轮) vs exp5(YOLOv8s+CBAM, 250轮) ← 轮次相同，最公平的对比')
print('  - exp4(YOLOv8s, 250轮) vs exp6(YOLOv8s+CBAM, 200轮) ← CBAM+迁移，轮次略少')
print('='*70)
