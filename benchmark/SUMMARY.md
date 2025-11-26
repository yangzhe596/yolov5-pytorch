# 基准测试工具总结

## 已创建的文件

```
benchmark/
├── __init__.py                      # Python 包初始化文件
├── training_benchmark.py            # 完整训练流程基准测试 (17KB)
├── dataloader_benchmark.py          # 数据加载详细分析 (20KB) ⭐ 新增
├── quick_benchmark.sh               # 快速训练测试 (50 batches)
├── quick_dataloader_benchmark.sh    # 快速数据加载测试 (100 samples) ⭐ 新增
├── full_benchmark.sh                # 完整测试 (多种配置对比)
├── README.md                        # 详细文档 (9.1KB)
├── USAGE.md                         # 快速使用指南 (4.6KB)
├── DATALOADER_ANALYSIS.md           # 数据加载分析文档 (12KB) ⭐ 新增
└── SUMMARY.md                       # 本文档
```

## 功能特性

### ✅ 已实现的功能

1. **详细的步骤计时**
   - 数据加载 (data_loading)
   - 数据传输 (data_transfer)
   - 前向传播 (forward)
   - 损失计算 (loss_compute)
   - 反向传播 (backward)
   - 优化器更新 (optimizer_step)

2. **性能指标**
   - 平均耗时、标准差、最小/最大值
   - 各步骤占比分析
   - 训练吞吐量 (samples/s)
   - GPU 内存使用情况
   - 预计训练时间

3. **自动分析**
   - 瓶颈识别
   - 优化建议生成
   - 配置对比

4. **灵活配置**
   - 支持 RGB/Event 模态
   - 可调整 batch_size
   - 可调整 DataLoader 参数
   - 可调整测试批次数

## 快速开始

### 1. 基本测试

```bash
/home/yz/.conda/envs/torch/bin/python3 benchmark/training_benchmark.py \
    --modality rgb \
    --num_batches 50
```

### 2. 快速测试

```bash
bash benchmark/quick_benchmark.sh
```

### 3. 完整测试

```bash
bash benchmark/full_benchmark.sh
```

## 测试结果示例

### 完整训练流程测试

基于初步测试（10 batches, batch_size=4）：

```
性能瓶颈分析:
- 数据加载: 41.4% ⚠️ 主要瓶颈
- 反向传播: 23.3%
- 前向传播: 5.7%
- 损失计算: 4.9%

训练吞吐量: 11.99 samples/s
预计 50 epochs: 111.56 小时
```

**优化建议**: 数据加载是主要瓶颈，建议增加 num_workers 和 prefetch_factor。

### 数据加载详细分析 ⭐ 新增

基于详细测试（100 samples）：

```
数据加载瓶颈细分:
- Mosaic 增强: 41.5% ⚠️ 最大瓶颈
- 图像读取 (PIL): 30.0%
- RGB 转换: 20.7%
- 图像缩放: 6.5%
- 其他: 1.3%

分组统计:
- 图像 I/O: 50.8%
- Mosaic 增强: 41.5%
- 图像缩放: 6.5%
- 颜色增强: 0.6%
- 预处理: 0.6%
```

**关键发现**:
1. **Mosaic 增强**是最大瓶颈（需要读取和处理 4 张图片）
2. **图像 I/O**占用一半时间（受磁盘速度影响）
3. 颜色增强和预处理不是瓶颈

**优化建议**:
1. 降低 Mosaic 概率（0.5 → 0.3）
2. 减少强增强轮次（0.7 → 0.5）
3. 使用 SSD 存储数据集
4. 考虑 cv2 替代 PIL

## 优化后的配置

已在 `config_fred.py` 中应用的优化：

```python
NUM_WORKERS = 8              # CPU 核心数的一半
PREFETCH_FACTOR = 4          # 每个 worker 预取 4 个 batch
PERSISTENT_WORKERS = True    # 保持 workers 存活
```

总预取量 = 8 × 4 = 32 batches

## 下一步优化方向

### 基于测试结果的优化建议

#### 1. 数据加载优化（最重要）⭐

**问题**: 数据加载占用 41.4%，其中 Mosaic 增强占 41.5%

**优化方案**:
```python
# config_fred.py
MOSAIC_PROB = 0.3              # 从 0.5 降低到 0.3
SPECIAL_AUG_RATIO = 0.5        # 从 0.7 降低到 0.5
NUM_WORKERS = 12               # 从 8 增加到 12
PREFETCH_FACTOR = 6            # 从 4 增加到 6
```

**预期效果**: 训练速度提升 30-50%

#### 2. 存储优化

**问题**: 图像 I/O 占用 50.8%

**优化方案**:
- 使用 SSD 存储数据集
- 考虑使用 cv2 替代 PIL

**预期效果**: I/O 时间减少 30-50%

#### 3. GPU 利用率优化

**问题**: 前向传播仅占 5.7%，GPU 可能未充分利用

**优化方案**:
- 增加 batch_size（如果内存允许）
- 启用混合精度训练（FP16）

**预期效果**: GPU 利用率提升 10-20%

#### 4. 其他优化

- 使用更快的图像缩放方法（BILINEAR 替代 BICUBIC）
- 降低 MixUp 概率
- 考虑使用 DALI 加速数据加载

## 使用文档

- **详细文档**: `benchmark/README.md`
- **快速指南**: `benchmark/USAGE.md`
- **项目文档**: `AGENTS.md`

## 性能基准参考

### RTX 3090 预期性能

| Batch Size | 吞吐量 (samples/s) | GPU 内存 | 每 epoch 时间 |
|-----------|-------------------|---------|--------------|
| 4         | 15-20             | 3-4 GB  | ~90 分钟     |
| 8         | 25-35             | 5-6 GB  | ~50 分钟     |
| 16        | 40-50             | 8-10 GB | ~35 分钟     |

*注: 基于 FRED RGB 数据集 (96,271 张训练图片)*

## 注意事项

1. ✅ 脚本会自动进行 5 个 batch 的预热
2. ✅ 所有 GPU 操作都会同步以确保计时准确
3. ✅ 支持命令行参数自定义配置
4. ✅ 自动生成优化建议
5. ⚠️ 建议至少测试 50 个 batch 以获得稳定统计
6. ⚠️ 测试时关闭其他占用 GPU 的程序

## 更新日志

### v1.0 (2025-11-02)
- ✅ 初始版本
- ✅ 支持详细的步骤计时
- ✅ 自动瓶颈分析
- ✅ 优化建议生成
- ✅ 支持 RGB/Event 模态
- ✅ 灵活的参数配置
- ✅ 完整的文档

## 相关优化

同时优化了 `train_fred.py` 和 `config_fred.py`：

1. ✅ 添加 `prefetch_factor` 参数
2. ✅ 添加 `persistent_workers` 参数
3. ✅ 优化 `num_workers` 配置
4. ✅ 确保 `pin_memory=True`

这些优化预计可提升训练速度 20-30%。
