# -*- coding: utf-8 -*-
import csv, math

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

s8 = summarize(read_csv(r'd:\文档\毕业设计\teeth\runs\exp8_yolov8s_250ep_scratch\results.csv'))
s5 = summarize(read_csv(r'd:\文档\毕业设计\teeth\runs\exp5_yolov8s_cbam_250ep\results.csv'))

print('='*60)
print('exp8  YOLOv8s（无CBAM）从零250轮 结果')
print('='*60)
print('总轮次:', s8['total_ep'])
print('最终: P={:.4f}  R={:.4f}  mAP50={:.4f}  mAP50-95={:.4f}'.format(
    s8['last_p'], s8['last_r'], s8['last_m50'], s8['last_m95']))
print('最佳 mAP50={:.4f} @Epoch {}  |  最佳 mAP50-95={:.4f} @Epoch {}'.format(
    s8['best_m50'], s8['best_m50_ep'], s8['best_m95'], s8['best_m95_ep']))
print('后50轮均值={:.4f}  标准差={:.5f}  ({})'.format(
    s8['mean50'], s8['std50'],
    '已收敛' if s8['std50'] < 0.002 else '基本收敛' if s8['std50'] < 0.003 else '仍震荡'))

print()
print('='*60)
print('控制变量对比：exp8 YOLOv8s vs exp5 YOLOv8s+CBAM')
print('（相同起点：yolov8s.pt，相同轮次：250轮）')
print('='*60)
print('{:22s} {:>12s} {:>14s} {:>10s} {:>10s}'.format('模型','最佳mAP50','最佳mAP50-95','末轮P','末轮R'))
print('-'*70)
print('{:22s} {:>12.4f} {:>14.4f} {:>10.4f} {:>10.4f}'.format(
    'exp8 YOLOv8s(无CBAM)', s8['best_m50'], s8['best_m95'], s8['last_p'], s8['last_r']))
print('{:22s} {:>12.4f} {:>14.4f} {:>10.4f} {:>10.4f}'.format(
    'exp5 YOLOv8s+CBAM',   s5['best_m50'], s5['best_m95'], s5['last_p'], s5['last_r']))
print()
diff50 = s5['best_m50'] - s8['best_m50']
diff95 = s5['best_m95'] - s8['best_m95']
print('CBAM 带来的变化（exp5 - exp8）:')
print('  mAP50:    {:+.4f} ({:+.2f}%)'.format(diff50, diff50*100))
print('  mAP50-95: {:+.4f} ({:+.2f}%)'.format(diff95, diff95*100))
if diff50 > 0:
    print('  结论: CBAM 有效，mAP50 提升了 {:.2f}%'.format(diff50*100))
else:
    print('  结论: CBAM 在此设置下未带来提升')
print('='*60)
