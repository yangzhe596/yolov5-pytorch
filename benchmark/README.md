# YOLOv5 训练性能基准测试

## 概述

这个目录包含用于测试 YOLOv5 训练性能的基准测试工具，可以详细分析训练过程中各个步骤的耗时，帮助识别性能瓶颈并优化训练速度。

## 功能特性

### 测试的关键步骤

1. **数据加载** (`data_loading`)
   - 从磁盘读取图片和标注
   - 数据预处理
   - Batch 组装

2. **数据传输** (`data_transfer`)
   - CPU 到 GPU 的数据传输
   - 使用 pinned memory 加速

3. **前向传播** (`forward`)
   - 模型推理计算
   - 特征提取和检测头计算

4. **损失计算** (`loss_compute`)
   - YOLOv5 损失函数计算
   - 包括分类、定位、置信度损失

5. **反向传播** (`backward`)
   - 梯度计算
   - 反向传播过程

6. **优化器更新** (`optimizer_step`)
   - 参数更新
   - 梯度应用

### 输出指标

- **平均耗时**: 每个步骤的平均执行时间
- **标准差**: 时间波动情况
- **最小/最大值**: 耗时范围
- **总计时间**: 累计耗时
- **占比分析**: 各步骤占总时间的百分比
- **吞吐量**: 每秒处理的样本数
- **GPU 内存**: 显存使用情况
- **瓶颈分析**: 自动识别性能瓶颈
- **优化建议**: 基于测试结果的优化建议

## 使用方法

### 1. 基本用法

```bash
# 测试 RGB 模态 (默认 100 batches)
python benchmark/training_benchmark.py --modality rgb

# 测试 Event 模态
python benchmark/training_benchmark.py --modality event

# 指定测试批次数
python benchmark/training_benchmark.py --modality rgb --num_batches 50
```

### 2. 自定义参数

```bash
# 自定义 batch size
python benchmark/training_benchmark.py --modality rgb --batch_size 16

# 自定义 DataLoader 参数
python benchmark/training_benchmark.py \
    --modality rgb \
    --num_workers 16 \
    --prefetch_factor 8 \
    --persistent_workers True

# 禁用 CUDA (CPU 测试)
python benchmark/training_benchmark.py --modality rgb --no_cuda
```

### 3. 快速测试

```bash
# 使用快捷脚本 (50 batches)
bash benchmark/quick_benchmark.sh
```

## 参数说明

### 必需参数

- `--modality`: 选择模态 (`rgb` 或 `event`)

### 可选参数

#### 测试参数
- `--num_batches`: 测试的批次数量 (默认: 100)
  - 更多批次 = 更准确的统计，但耗时更长
  - 建议: 50-100 批次用于快速测试，200+ 批次用于详细分析

- `--batch_size`: 批次大小 (默认: 使用配置文件中的值)
  - 影响 GPU 利用率和内存占用
  - 建议: 8-16 (根据 GPU 显存调整)

#### DataLoader 参数
- `--num_workers`: DataLoader workers 数量 (默认: 8)
  - 影响数据加载速度
  - 建议: CPU 核心数的一半

- `--prefetch_factor`: 预取因子 (默认: 4)
  - 每个 worker 预取的 batch 数量
  - 建议: 2-8

- `--persistent_workers`: 是否使用持久化 workers (默认: True)
  - 保持 workers 在 epoch 间存活
  - 建议: 启用以减少启动开销

#### 设备参数
- `--cuda`: 使用 CUDA (默认: True)
- `--no_cuda`: 禁用 CUDA

## 输出示例

```
================================================================================
性能基准测试结果
================================================================================

步骤                   平均(ms)      标准差(ms)    最小(ms)      最大(ms)      总计(s)    
--------------------------------------------------------------------------------
train_forward             45.23        2.15        42.10        52.30        4.52
train_data_loading        28.67        5.43        22.15        45.80        2.87
train_backward            22.45        1.87        19.30        28.90        2.25
train_loss_compute        12.34        0.98        10.50        15.20        1.23
train_data_transfer        8.56        1.23         6.80        12.40        0.86
train_optimizer_step       5.67        0.45         4.90         7.20        0.57
--------------------------------------------------------------------------------
总计                                                                          12.30

步骤                   占比(%)       吞吐量(samples/s)
--------------------------------------------------------------------------------
train_forward             36.7              22.13
train_data_loading        23.3              34.88
train_backward            18.3              44.49
train_loss_compute        10.0              81.10
train_data_transfer        7.0             116.67
train_optimizer_step       4.6             176.37

================================================================================
额外性能指标
================================================================================

平均每批次时间: 123.45 ms
训练吞吐量: 64.80 samples/s
预计每 epoch 时间: 3.50 分钟
预计 50 epochs 总时间: 2.92 小时

GPU 内存使用:
  已分配: 4.23 GB
  已缓存: 5.67 GB
  峰值: 5.89 GB

================================================================================
瓶颈分析
================================================================================

主要耗时步骤 (占比 > 5%):
  - train_forward: 36.7% (平均 45.23 ms)
  - train_data_loading: 23.3% (平均 28.67 ms)
  - train_backward: 18.3% (平均 22.45 ms)
  - train_loss_compute: 10.0% (平均 12.34 ms)
  - train_data_transfer: 7.0% (平均 8.56 ms)

================================================================================
优化建议
================================================================================

数据加载占比: 23.3%
模型计算占比: 36.7%

⚠️  数据加载成为瓶颈，建议:
  1. 增加 num_workers (当前: 8)
  2. 增加 prefetch_factor (当前: 4)
  3. 使用更快的存储设备 (SSD)
  4. 减少数据增强的复杂度
```

## 性能优化指南

### 1. 数据加载优化

如果 `data_loading` 占比 > 20%:

```bash
# 增加 workers 和预取因子
python benchmark/training_benchmark.py \
    --modality rgb \
    --num_workers 16 \
    --prefetch_factor 8
```

**优化措施**:
- 增加 `num_workers` (建议: 8-16)
- 增加 `prefetch_factor` (建议: 4-8)
- 启用 `persistent_workers`
- 使用 SSD 存储数据集
- 减少数据增强复杂度

### 2. 模型计算优化

如果 `forward` + `backward` 占比 > 60%:

```bash
# 使用更大的 batch size
python benchmark/training_benchmark.py \
    --modality rgb \
    --batch_size 16
```

**优化措施**:
- 增加 `batch_size` (提高 GPU 利用率)
- 启用混合精度训练 (FP16)
- 使用更小的模型 (如 YOLOv5-s)
- 减小输入尺寸

### 3. 数据传输优化

如果 `data_transfer` 占比 > 10%:

**优化措施**:
- 确保启用 `pin_memory=True`
- 使用 CUDA 流 (streams)
- 减小 batch size (减少传输量)

### 4. 内存优化

如果 GPU 内存不足:

**优化措施**:
- 减小 `batch_size`
- 减小 `input_shape`
- 减少 `prefetch_factor`
- 使用梯度累积

## 对比测试

### 测试不同配置的影响

```bash
# 配置 1: 默认配置
python benchmark/training_benchmark.py --modality rgb --num_batches 100

# 配置 2: 增加 workers
python benchmark/training_benchmark.py --modality rgb --num_batches 100 --num_workers 16

# 配置 3: 增加预取
python benchmark/training_benchmark.py --modality rgb --num_batches 100 --prefetch_factor 8

# 配置 4: 增加 batch size
python benchmark/training_benchmark.py --modality rgb --num_batches 100 --batch_size 16
```

### 对比不同模态

```bash
# RGB 模态
python benchmark/training_benchmark.py --modality rgb --num_batches 100

# Event 模态
python benchmark/training_benchmark.py --modality event --num_batches 100
```

## 注意事项

1. **预热**: 脚本会自动进行 5 个 batch 的预热，避免初始化开销影响测试结果

2. **CUDA 同步**: 所有 GPU 操作都会进行同步，确保计时准确

3. **测试环境**: 
   - 关闭其他占用 GPU 的程序
   - 确保数据集已缓存到内存/SSD
   - 避免在测试期间进行其他 I/O 操作

4. **统计准确性**: 
   - 更多批次 = 更准确的统计
   - 建议至少测试 50 个批次
   - 多次运行取平均值

5. **配置一致性**: 
   - 使用与实际训练相同的配置
   - 确保数据增强设置一致

## 故障排除

### 问题 1: 数据集未找到

```
FileNotFoundError: 训练集标注文件不存在
```

**解决方案**: 先运行数据集转换脚本
```bash
python convert_fred_to_coco_v2.py --modality rgb
```

### 问题 2: CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**: 减小 batch size
```bash
python benchmark/training_benchmark.py --modality rgb --batch_size 4
```

### 问题 3: DataLoader 卡住

**可能原因**: 
- `num_workers` 过高
- 磁盘 I/O 瓶颈

**解决方案**: 减少 workers
```bash
python benchmark/training_benchmark.py --modality rgb --num_workers 4
```

## 文件说明

- `training_benchmark.py`: 完整训练流程基准测试脚本
- `dataloader_benchmark.py`: 数据加载详细分析脚本 ⭐ 新增
- `quick_benchmark.sh`: 快速训练测试脚本 (50 batches)
- `quick_dataloader_benchmark.sh`: 快速数据加载测试脚本 (100 samples) ⭐ 新增
- `full_benchmark.sh`: 完整测试脚本 (多种配置对比)
- `README.md`: 本文档
- `USAGE.md`: 快速使用指南
- `SUMMARY.md`: 总结文档

## 相关文档

- `../config_fred.py`: 训练配置文件
- `../train_fred.py`: 训练脚本
- `../AGENTS.md`: 项目完整文档

## 更新日志

### v1.0 (2025-11-02)
- 初始版本
- 支持详细的步骤计时
- 自动瓶颈分析
- 优化建议生成
