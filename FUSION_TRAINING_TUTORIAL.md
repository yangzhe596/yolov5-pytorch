# FRED Fusion 训练教程

## 概述

`train_fred_fusion.py` 是 FRED 数据集的双模态融合训练脚本。它支持同时训练 RGB 和 Event 两种模态，通过特征融合提升检测性能。

## 主要特性

### 1. 双模态融合架构
- 支持 RGB + Event 双模态并行处理
- 相同的 backbone 网络（保证特征空间一致性）
- 特征拼接 + 1×1 卷积压缩（平衡性能和计算量）

### 2. 融合增强策略
- **融合标注文件**: 包含配对状态和时间戳信息
- **自动过滤低质量配对**: 支持过滤非配对样本
- **跨模态 Mosaic**: 双模态同步进行数据增强
- **Discriminator Loss**: 可选的配对质量判别器

### 3. 灵活的训练配置
- 模态选择: RGB / Event / Dual（双模态）
- 融合模式: Concat / Add / Channel Attention
- 压缩比率: 可调节特征通道压缩比例
- 融合阈值: 控制配对样本的时间差阈值

## 快速开始

### 1. 训练双模态融合模型

```bash
# 方法1: 使用 python 执行
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py

# 方法2: 使用 bash 脚本
bash start_training_fusion.sh
```

### 2. 只使用 RGB 模态

```bash
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --modality rgb
```

### 3. 只使用 Event 模态

```bash
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --modality event
```

### 4. 快速验证（仅 100 个 batch）

```bash
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --quick_test
```

### 5. 仅评估模式（不训练）

```bash
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --eval_only
```

### 6. 只进行冻结训练

```bash
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --freeze_training
```

## 配置参数

### 核心配置（在 `config_fred.py` 中）

```python
# 模态选择
MODALITY = 'dual'  # 'dual', 'rgb', 'event'

# 融合模式
FUSION_MODE = 'concat'  # 'concat', 'add', 'channel_attn'

# 压缩比率
COMPRESSION_RATIO = 0.5  # 将通道数压缩到一半

# 融合阈值
FUSION_THRESHOLD = 0.1  # 最大时间差（秒）

# 使用融合信息
USE_FUSION_INFO = True

# 训练配置
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_BATCH_SIZE = 8  # 由于双流输入，batch size 需要减少
UNFREEZE_BATCH_SIZE = 4
```

### 训练流程

#### 第一阶段：冻结训练 (0-50 Epoch)

```
Frozen stage (0-50 Epoch)
  - 冻结主干网络（RGB 和 Event 的 backbone）
  - 仅训练检测头
  - 显存占用小 (~6GB)
  - 每个 epoch 约 8 分钟
  - 特点：快速收敛，适合显存不足的场景
```

#### 第二阶段：解冻训练 (50-300 Epoch)

```
Unfreeze stage (51-300 Epoch)
  - 解冻所有层，全网络训练
  - 双流 backbone 同时更新
  - 显存占用大 (~14GB)
  - 每个 epoch 约 15 分钟
  - 特点：提升性能，适合追求最佳效果
```

**注意**: 双模态融合训练的显存占用是单模态的 2 倍左右！

## 输出文件结构

```
logs/fred_fusion/
├── loss_2025_10_23_XX_XX_XX/
│   ├── events.out.tfevents.*
│   ├── epoch_loss.png
│   ├── val_loss.png
│   ├── lr.png
│   └── fusion_stats.csv          # 融合信息统计
├── .temp_map_out/                # 评估结果
│   └── results/
│       └── results.txt
├── fred_fusion_best.pth          # 最佳模型
├── fred_fusion_final.pth         # 最终模型
└── ep*-loss*-val_loss*.pth       # 定期保存的模型
```

## 融合信息统计

训练过程中会实时显示融合相关信息：

```
融合信息: Dual Rate: 5120/13629 (37.6%), Avg Time Diff: 15.23ms
```

- **Dual Rate**: 配对样本比例（RGB 和 Event 时间戳在阈值内）
- **Avg Time Diff**: 平均时间差（从融合标注文件中读取）

这些信息可以帮助你评估融合数据集的质量。

## 网络架构参数

### 基础架构
- **backbone**: CSPDarknet (默认)
- **融合位置**: 在 backbone 的三个特征层（C3, C4, C5）后进行
- **融合方式**: 
  1. RGB 和 Event 分别通过相同的 backbone
  2. 在每个特征层拼接特征图
  3. 使用 1×1 卷积压缩通道数（默认压缩到一半）

### 特征层
```
输入: RGB [3, 640, 640] + Event [3, 640, 640]
↓
Backbone (CSPDarknet) → F1 [256, 80, 80] + F1 [256, 80, 80]
                    → F2 [512, 40, 40] + F2 [512, 40, 40]
                    → F3 [1024, 20, 20] + F3 [1024, 20, 20]
↓
融合: Concat → [512, 80, 80], [1024, 40, 40], [2048, 20, 20]
↓
压缩: 1×1 Conv → [256, 80, 80], [512, 40, 40], [1024, 20, 20]
↓
后续处理: SPPF + Neck + Head
```

## 常见问题

### Q1: 显存不足怎么办？

```bash
# 解决方案1: 减小 batch size
# 在 config_fred.py 中设置
UNFREEZE_BATCH_SIZE = 2  # 从 4 降到 2

# 解决方案2: 只进行冻结训练
/home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py --freeze_training

# 解决方案3: 使用更小的模型
PHI = 's'  # 使用 yolov5s，而不是默认的 yolov5m
```

### Q2: 如何提升融合效果？

```bash
# 提升方案1: 调整融合阈值（找到最佳配对窗口）
FUSION_THRESHOLD = 0.05  # 降低阈值，使用更多配对样本

# 提升方案2: 调整压缩比率
COMPRESSION_RATIO = 0.75  # 保留更多信息，但显存占用增加

# 提升方案3: 尝试不同的融合模式
FUSION_MODE = 'channel_attn'  # 使用注意力机制
```

### Q3: 评估结果包括 mAP 吗？

不包括。由于双模态融合模型评估 mAP 会导致内存溢出，评估脚本只计算验证损失。如果需要 mAP 评估，可以：

```bash
# 方法1: 使用单模态模型分别评估
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality rgb
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality event

# 方法2: 手动提取特征进行评估（高级）
```

## 预期性能

### FRED Fusion RGB + Event (YOLOv5m)
- **显存占用**: 冻结阶段 ~6GB, 解冻阶段 ~14GB
- **训练时间**: ~60 小时 (300 epochs)
- **验证 loss**: 预期比单模态低 15-20%
- **实际检测性能**: 需要通过预测脚本验证

### 对比单模态
- **RGB 单模态**: ~45 小时, 6GB/8GB 显存
- **Event 单模态**: ~45 小时, 6GB/8GB 显存
- **Fusion 双模态**: ~60 小时, 6GB/14GB 显存
- **性能增益**: 预计 5-15% mAP 提升（需要验证）

## 监控训练

### 方法1: 实时查看终端输出

训练过程会显示：
- 当前 epoch 和进度
- 训练损失和验证损失
- 学习率
- 融合统计信息
- 每个 epoch 的耗时

### 方法2: TensorBoard

```bash
# 标准训练
tensorboard --logdir logs/fred_fusion/

# 浏览器访问
http://localhost:6006
```

### 方法3: 查看日志文件

```bash
# 查看最新的训练日志
ls -lt logs/fred_fusion/

# 查看融合统计
cat logs/fred_fusion/loss_*/fusion_stats.csv
```

## 下一步

1. **训练完成后**: 预测测试
   ```bash
   /home/yz/.conda/envs/torch/bin/python3 predict_fred_fusion.py
   ```

2. **对比单模态性能**: 分析融合是否带来提升

3. **调整超参数**: 根据结果优化融合策略

4. **导出 ONNX**: 部署到生产环境

---

**文档创建时间**: 2025-11-24  
**项目路径**: `/mnt/data/code/yolov5-pytorch`  
**最新更新**: 完成了训练脚本的完整实现