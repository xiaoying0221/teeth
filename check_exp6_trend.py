# -*- coding: utf-8 -*-
import csv

rows = []
with open(r'd:\文档\毕业设计\teeth\runs\exp6_yolov8s_cbam_transfer\results.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

epochs = [int(r['epoch']) for r in rows]
m50 = [float(r['metrics/mAP50(B)']) for r in rows]
val_box = [float(r['val/box_loss']) for r in rows]

# 最后20轮
print('最后20轮 mAP@50 趋势:')
for i in range(-20, 0):
    trend = '+' if i > -20 and m50[i] >= m50[i-1] else '-'
    print('  Epoch {:3d}: mAP50={:.4f}  val/box_loss={:.4f}  {}'.format(epochs[i], m50[i], val_box[i], trend))

import math
last20 = m50[-20:]
mean = sum(last20)/len(last20)
std = math.sqrt(sum((x-mean)**2 for x in last20)/len(last20))
print('\n后20轮 mAP50 均值: {:.4f}'.format(mean))
print('后20轮 mAP50 标准差: {:.5f}'.format(std))
print('趋势: {}'.format('已收敛' if std < 0.002 else '仍在上升' if m50[-1] > m50[-10] else '震荡中'))
print('最后10轮均值 vs 前10轮均值: {:.4f} vs {:.4f}'.format(
    sum(m50[-10:])/10, sum(m50[-20:-10])/10))
