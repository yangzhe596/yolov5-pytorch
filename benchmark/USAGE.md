# 基准测试快速使用指南

## 快速开始

### 1. 基本测试（推荐）

测试 RGB 模态，50 个批次：

```bash
/home/yz/miniforge3/envs/torch/bin/python3 benchmark/training_benchmark.py \
    --modality rgb \
    --num_batches 50
```

### 2. 快速测试

使用快捷脚本（10 个批次）：

```bash
bash benchmark/quick_benchmark.sh
```

### 3. 完整测试

测试多种配置对比：

```bash
bash benchmark/full_benchmark.sh
```

## 常用命令

### 测试不同模态

```bash
# RGB 模态
python benchmark/training_benchmark.py --modality rgb --num_batches 50

# Event 模态
python benchmark/training_benchmark.py --modality event --num_batches 50
```

### 测试不同 batch size

```bash
# Batch size = 4
python benchmark/training_benchmark.py --modality rgb --batch_size 4 --num_batches 50

# Batch size = 8 (默认)
python benchmark/training_benchmark.py --modality rgb --batch_size 8 --num_batches 50

# Batch size = 16
python benchmark/training_benchmark.py --modality rgb --batch_size 16 --num_batches 50
```

### 测试不同 DataLoader 配置

```bash
# 增加 workers
python benchmark/training_benchmark.py --modality rgb --num_workers 16 --num_batches 50

# 增加 prefetch_factor
python benchmark/training_benchmark.py --modality rgb --prefetch_factor 8 --num_batches 50

# 组合优化
python benchmark/training_benchmark.py --modality rgb \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_batches 50
```

## 输出解读

### 关键指标

1. **数据加载占比** (`train_data_loading`)
   - **正常**: < 20%
   - **需优化**: > 30%
   - **优化方法**: 增加 num_workers, prefetch_factor

2. **模型计算占比** (`train_forward` + `train_backward`)
   - **正常**: 50-70%
   - **GPU 利用不足**: < 40%
   - **优化方法**: 增加 batch_size, 使用 FP16

3. **训练吞吐量** (samples/s)
   - **RTX 3090 参考值**:
     - Batch 4: ~15-20 samples/s
     - Batch 8: ~25-35 samples/s
     - Batch 16: ~40-50 samples/s

4. **GPU 内存使用**
   - **Batch 4**: ~3-4 GB
   - **Batch 8**: ~5-6 GB
   - **Batch 16**: ~8-10 GB

### 瓶颈识别

脚本会自动识别瓶颈并给出建议：

- **数据加载瓶颈**: 增加 workers 和 prefetch
- **计算瓶颈**: 增加 batch size 或使用 FP16
- **内存瓶颈**: 减小 batch size 或 input shape

## 性能优化流程

### Step 1: 运行基准测试

```bash
python benchmark/training_benchmark.py --modality rgb --num_batches 50
```

### Step 2: 查看瓶颈分析

查看输出中的"瓶颈分析"和"优化建议"部分。

### Step 3: 应用优化

根据建议调整参数，例如：

```bash
# 如果数据加载是瓶颈
python benchmark/training_benchmark.py --modality rgb \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_batches 50
```

### Step 4: 对比结果

对比优化前后的吞吐量和各步骤耗时。

### Step 5: 更新配置文件

将最优配置更新到 `config_fred.py`：

```python
NUM_WORKERS = 16
PREFETCH_FACTOR = 8
PERSISTENT_WORKERS = True
```

## 示例输出

```
================================================================================
性能基准测试结果
================================================================================

步骤                   平均(ms)       占比(%)    
--------------------------------------------------------------------------------
train_data_loading       163.78       41.4%     ⚠️ 瓶颈
train_backward           101.29       23.3%
train_forward             24.66        5.7%
train_loss_compute        21.54        4.9%
train_optimizer_step       4.94        1.1%
train_data_transfer        1.16        0.3%

训练吞吐量: 11.99 samples/s
预计每 epoch 时间: 133.87 分钟
预计 50 epochs 总时间: 111.56 小时

优化建议:
⚠️  数据加载成为瓶颈，建议:
  1. 增加 num_workers (当前: 8)
  2. 增加 prefetch_factor (当前: 4)
```

## 注意事项

1. **测试环境**: 确保没有其他程序占用 GPU
2. **数据集位置**: 使用 SSD 存储数据集可显著提升性能
3. **批次数量**: 至少测试 50 个批次以获得稳定统计
4. **多次测试**: 建议运行 2-3 次取平均值

## 故障排除

### 问题: 数据加载很慢

```bash
# 检查磁盘 I/O
iostat -x 1

# 增加 workers
python benchmark/training_benchmark.py --modality rgb --num_workers 16
```

### 问题: GPU 利用率低

```bash
# 监控 GPU
nvidia-smi -l 1

# 增加 batch size
python benchmark/training_benchmark.py --modality rgb --batch_size 16
```

### 问题: 内存不足

```bash
# 减小 batch size
python benchmark/training_benchmark.py --modality rgb --batch_size 4
```
