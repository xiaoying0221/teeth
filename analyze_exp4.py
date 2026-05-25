import csv
import math

# ==================== 读取数据 ====================
def read_results(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows

exp4 = read_results(r'd:\文档\毕业设计\teeth\runs\exp4_yolov8s_250ep\results.csv')

# ==================== 提取关键指标 ====================
epochs      = [int(r['epoch'])                          for r in exp4]
map50       = [float(r['metrics/mAP50(B)'])             for r in exp4]
map5095     = [float(r['metrics/mAP50-95(B)'])          for r in exp4]
precision   = [float(r['metrics/precision(B)'])         for r in exp4]
recall      = [float(r['metrics/recall(B)'])            for r in exp4]
train_box   = [float(r['train/box_loss'])               for r in exp4]
train_cls   = [float(r['train/cls_loss'])               for r in exp4]
train_dfl   = [float(r['train/dfl_loss'])               for r in exp4]
val_box     = [float(r['val/box_loss'])                 for r in exp4]
val_cls     = [float(r['val/cls_loss'])                 for r in exp4]
val_dfl     = [float(r['val/dfl_loss'])                 for r in exp4]

# ==================== 关键统计 ====================
best_map50_idx   = map50.index(max(map50))
best_map5095_idx = map5095.index(max(map5095))
best_prec_idx    = precision.index(max(precision))
best_rec_idx     = recall.index(max(recall))

print('=' * 60)
print('  YOLOv8s 250 Epoch 训练结果分析报告')
print('=' * 60)

print('\n【最终轮次 (Epoch 250) 指标】')
last = exp4[-1]
print(f'  Precision  : {float(last["metrics/precision(B)"]):.4f}')
print(f'  Recall     : {float(last["metrics/recall(B)"]):.4f}')
print(f'  mAP@50     : {float(last["metrics/mAP50(B)"]):.4f}')
print(f'  mAP@50-95  : {float(last["metrics/mAP50-95(B)"]):.4f}')
print(f'  val/box_loss: {float(last["val/box_loss"]):.5f}')
print(f'  val/cls_loss: {float(last["val/cls_loss"]):.5f}')
print(f'  val/dfl_loss: {float(last["val/dfl_loss"]):.5f}')

print('\n【最佳指标】')
print(f'  最高 mAP@50     : {max(map50):.4f}  @ Epoch {epochs[best_map50_idx]}')
print(f'  最高 mAP@50-95  : {max(map5095):.4f}  @ Epoch {epochs[best_map5095_idx]}')
print(f'  最高 Precision  : {max(precision):.4f}  @ Epoch {epochs[best_prec_idx]}')
print(f'  最高 Recall     : {max(recall):.4f}  @ Epoch {epochs[best_rec_idx]}')

print('\n【训练损失变化】')
print(f'  box_loss  : Epoch1={train_box[0]:.5f}  →  Epoch250={train_box[-1]:.5f}  (下降 {train_box[0]-train_box[-1]:.5f})')
print(f'  cls_loss  : Epoch1={train_cls[0]:.5f}  →  Epoch250={train_cls[-1]:.5f}  (下降 {train_cls[0]-train_cls[-1]:.5f})')
print(f'  dfl_loss  : Epoch1={train_dfl[0]:.5f}  →  Epoch250={train_dfl[-1]:.5f}  (下降 {train_dfl[0]-train_dfl[-1]:.5f})')

print('\n【验证损失变化】')
print(f'  box_loss  : Epoch1={val_box[0]:.5f}  →  Epoch250={val_box[-1]:.5f}  (下降 {val_box[0]-val_box[-1]:.5f})')
print(f'  cls_loss  : Epoch1={val_cls[0]:.5f}  →  Epoch250={val_cls[-1]:.5f}  (下降 {val_cls[0]-val_cls[-1]:.5f})')
print(f'  dfl_loss  : Epoch1={val_dfl[0]:.5f}  →  Epoch250={val_dfl[-1]:.5f}  (下降 {val_dfl[0]-val_dfl[-1]:.5f})')

print('\n【与历史实验对比 (mAP@50)】')
exp_data = [
    ('exp1', 'YOLOv8n', 50,  0.7366, 0.3843),
    ('exp2', 'YOLOv8n', 100, 0.7805, 0.4903),
    ('exp3', 'YOLOv8s', 100, 0.7995, 0.5313),
    ('exp4', 'YOLOv8s', 250, float(last['metrics/mAP50(B)']), float(last['metrics/mAP50-95(B)'])),
]
print(f'  {"实验":<6} {"模型":<10} {"Epochs":<8} {"mAP@50":<10} {"mAP@50-95"}')
print('  ' + '-'*50)
for name, model, ep, m50, m5095 in exp_data:
    print(f'  {name:<6} {model:<10} {ep:<8} {m50:<10.4f} {m5095:.4f}')

print('\n【收敛分析】')
# 计算后50轮的mAP50方差来判断是否收敛
last50 = map50[-50:]
mean50 = sum(last50) / len(last50)
var50  = sum((x - mean50) ** 2 for x in last50) / len(last50)
std50  = math.sqrt(var50)
print(f'  后50轮 mAP@50 均值   : {mean50:.4f}')
print(f'  后50轮 mAP@50 标准差 : {std50:.5f}')
print(f'  收敛判断             : {"已收敛 (std<0.003)" if std50 < 0.003 else "未完全收敛"}')

print('\n【结论】')
print(f'  1. 相比 exp3 (YOLOv8s 100ep), mAP@50 提升 {(float(last["metrics/mAP50(B)"])-0.7995)*100:.2f}%')
print(f'  2. 相比 exp3 (YOLOv8s 100ep), mAP@50-95 提升 {(float(last["metrics/mAP50-95(B)"])-0.5313)*100:.2f}%')
print(f'  3. 训练总时长约 {float(last["time"])/3600:.2f} 小时')
print('  4. 250轮训练带来了稳定的性能提升，损失函数持续下降')
print('=' * 60)
