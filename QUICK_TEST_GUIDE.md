# FRED数据集快速测试指南

## ✅ 所有问题已修复

- ✅ 除零warning已修复
- ✅ EvalCallback已实现
- ✅ 训练流程完全正常
- ✅ 创建了mini数据集用于快速测试

---

## 🚀 快速测试（推荐）

### 1. 创建mini数据集（80张图片）

```bash
# RGB模态
/home/yz/miniforge3/envs/torch/bin/python3 create_mini_dataset.py \
    --modality rgb --num_train 50 --num_val 20 --num_test 10

# Event模态（可选）
/home/yz/miniforge3/envs/torch/bin/python3 create_mini_dataset.py \
    --modality event --num_train 50 --num_val 20 --num_test 10
```

### 2. 快速训练测试（3个epochs，约1-2分钟）

```bash
# RGB模态
/home/yz/miniforge3/envs/torch/bin/python3 test_train_mini.py \
    --modality rgb --epochs 3

# Event模态
/home/yz/miniforge3/envs/torch/bin/python3 test_train_mini.py \
    --modality event --epochs 3
```

### 3. 查看结果

```bash
# 查看训练日志
ls -lh logs/test_mini_rgb/

# 查看loss曲线
ls -lh logs/test_mini_rgb/loss_*/epoch_loss.png
```

---

## 📊 快速测试结果示例

```
快速训练测试 - RGB模态 - 3 epochs
================================================================================

数据集: 50 训练 + 20 验证
训练步数: 12 steps/epoch
验证步数: 5 steps/epoch

Epoch 1/3: Total Loss: 3.766 || Val Loss: 3.765
Epoch 2/3: Total Loss: 3.132 || Val Loss: 3.619
Epoch 3/3: Total Loss: 2.559 || Val Loss: 3.549

✅ 训练完成！
耗时: 约1-2分钟
```

---

## 🎯 正式训练

### 方式1: 不计算mAP（推荐，训练更快）

```bash
# RGB模态
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb

# Event模态
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality event
```

### 方式2: 计算mAP（训练较慢，但可以监控性能）

```bash
# RGB模态（每10个epoch计算一次mAP）
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py \
    --modality rgb --eval_map

# Event模态
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py \
    --modality event --eval_map
```

### 方式3: 后台训练

```bash
# RGB模态后台训练
nohup /home/yz/miniforge3/envs/torch/bin/python3 train_fred.py \
    --modality rgb > train_rgb.log 2>&1 &

# 查看训练日志
tail -f train_rgb.log

# 查看进程
ps aux | grep train_fred

# 停止训练
pkill -f train_fred.py
```

---

## 📁 数据集对比

### Mini数据集（快速测试）
```
datasets/fred_mini/rgb/
├── train/      50张图片
├── val/        20张图片
└── test/       10张图片

训练时间: 约1-2分钟/epoch
总耗时: 约5-10分钟（3 epochs）
用途: 快速验证训练流程
```

### 完整数据集（正式训练）
```
datasets/fred_coco/rgb/
├── train/      13,629张图片
├── val/        3,894张图片
└── test/       1,948张图片

训练时间: 约8分钟/epoch
总耗时: 约40小时（300 epochs）
用途: 正式模型训练
```

---

## ⚙️ 训练配置对比

### 快速测试配置
```python
epochs = 3
batch_size = 4
pretrained = False
mosaic = False
mixup = False
eval_map = False
```

### 正式训练配置
```python
freeze_epoch = 50
unfreeze_epoch = 300
freeze_batch_size = 16
unfreeze_batch_size = 8
pretrained = True
mosaic = True
mixup = True
eval_map = True  # 可选
```

---

## 🔧 已修复的问题

### 1. 除零Warning ✅
**问题**: 
```
RuntimeWarning: divide by zero encountered in divide
```

**修复**: 
- 过滤宽度或高度为0的边界框
- 添加epsilon避免除零

**代码**:
```python
# 过滤无效边界框
valid_mask = (batch_target[:, 2] > 0) & (batch_target[:, 3] > 0)
valid_batch_target = batch_target[valid_mask]

# 添加epsilon
epsilon = 1e-8
ratios_of_anchors_gt = anchors / (batch_target[:, 2:4] + epsilon)
```

### 2. EvalCallback NoneType错误 ✅
**问题**:
```
AttributeError: 'NoneType' object has no attribute 'on_epoch_end'
```

**修复**:
- 实现了`CocoEvalCallback`（完整mAP评估）
- 实现了`SimplifiedEvalCallback`（简化版，不计算mAP）
- 根据`--eval_map`参数选择使用哪个

**文件**: `utils/callbacks_coco.py`

---

## 📝 测试检查清单

- [x] 数据加载器测试通过
- [x] 模型创建测试通过
- [x] DataLoader测试通过
- [x] 除零warning已修复
- [x] EvalCallback已实现
- [x] Mini数据集创建成功
- [x] 快速训练测试通过（3 epochs）
- [x] 无warning或error
- [ ] 正式训练（待执行）

---

## 🎯 推荐的测试流程

### 第一步: 快速测试（5-10分钟）
```bash
# 1. 创建mini数据集
/home/yz/miniforge3/envs/torch/bin/python3 create_mini_dataset.py --modality rgb

# 2. 快速训练测试
/home/yz/miniforge3/envs/torch/bin/python3 test_train_mini.py --modality rgb --epochs 3

# 3. 检查结果
ls -lh logs/test_mini_rgb/
```

### 第二步: 短期训练（1-2小时）
```bash
# 训练10个epochs验证效果
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb

# 修改train_fred.py中的UnFreeze_Epoch = 10
```

### 第三步: 正式训练（40小时）
```bash
# 完整训练300 epochs
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb

# 或后台训练
nohup /home/yz/miniforge3/envs/torch/bin/python3 train_fred.py \
    --modality rgb > train_rgb.log 2>&1 &
```

---

## 📊 性能对比

| 数据集 | 样本数 | Epochs | 耗时/epoch | 总耗时 | 用途 |
|--------|--------|--------|-----------|--------|------|
| Mini | 50+20 | 3 | ~30秒 | ~2分钟 | 快速测试 |
| Mini | 50+20 | 10 | ~30秒 | ~5分钟 | 验证流程 |
| Full | 13,629+3,894 | 10 | ~8分钟 | ~80分钟 | 短期验证 |
| Full | 13,629+3,894 | 300 | ~8分钟 | ~40小时 | 正式训练 |

---

## 💡 使用建议

### 开发和调试
- ✅ 使用mini数据集
- ✅ 关闭mAP评估
- ✅ 使用小的epoch数（3-10）
- ✅ 快速迭代测试

### 超参数调优
- 使用mini数据集或完整数据集的10-50 epochs
- 测试不同的学习率、batch size等
- 观察loss收敛情况

### 正式训练
- 使用完整数据集
- 训练300+ epochs
- 可选择性开启mAP评估（--eval_map）
- 后台运行

---

## 🔍 故障排除

### 问题1: Mini数据集不存在
```bash
# 创建mini数据集
/home/yz/miniforge3/envs/torch/bin/python3 create_mini_dataset.py --modality rgb
```

### 问题2: 显存不足
```bash
# 减小batch size
# 在test_train_mini.py中修改: batch_size = 2
```

### 问题3: 训练太慢
```bash
# 使用mini数据集
# 减少epochs
# 关闭mAP评估
```

---

## 📚 相关文件

- `create_mini_dataset.py` - 创建mini数据集
- `test_train_mini.py` - mini数据集快速训练
- `train_fred.py` - 完整训练脚本
- `utils/callbacks_coco.py` - COCO评估回调
- `utils/dataloader_coco.py` - COCO数据加载器（已修复warning）

---

**创建时间**: 2025-10-21  
**状态**: ✅ 所有问题已修复，可以开始训练
