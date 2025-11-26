# simple 模式使用指南

## 概述

simple 模式是 `convert_fred_to_fusion_v2.py` 的一个快速测试模式，它通过采样部分数据来加速数据集生成过程，适合快速验证和调试。

## 功能特点

- **按序列采样**：simple 模式在生成数据集之前，按比例随机采样部分视频序列
- **随机种子固定**：使用固定随机种子（42），确保采样结果可重现
- **采样比例可调**：默认 10%，可通过参数调整
- **不影响数据质量**：采样后的数据集结构和完整数据集完全相同

## 使用方法

### 基本用法

```bash
# 启用 simple 模式（默认采样 10% 数据）
python convert_fred_to_fusion_v2.py --split-mode frame --simple

# 自定义采样比例（5% 数据，适合更快的测试）
python convert_fred_to_fusion_v2.py --split-mode frame --simple --simple-ratio 0.05

# 完整数据集（禁用 simple 模式）
python convert_fred_to_fusion_v2.py --split-mode frame
```

### 进阶用法

```bash
# Simple 模式 + 自定义时间容差
python convert_fred_to_fusion_v2.py --split-mode frame --simple --threshold 0.02

# Simple 模式 + 自定义划分比例
python convert_fred_to_fusion_v2.py --split-mode frame --simple \
  --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# 仅生成部分数据用于快速验证
python convert_fred_to_fusion_v2.py --split-mode frame --simple --simple-ratio 0.01
```

## 参数说明

### `--simple`

启用简化模式。启用后，会从所有序列中随机采样部分序列用于生成数据集。

- **类型**：`action='store_true'`
- **默认值**：`False`
- **说明**：只在测试和调试时启用

### `--simple-ratio`

简化模式下的采样比例。

- **类型**：`float`
- **默认值**：`0.1`（10%）
- **范围**：`0.0 ~ 1.0`
- **说明**：
  - `0.1` = 采样 10% 的序列
  - `0.5` = 采样 50% 的序列
  - `1.0` = 采样所有序列（等同于停止 simple 模式）

## 工作原理

### 采样流程

```
1. 获取所有序列列表（51 个序列）
2. 计算采样数量：n_sample = int(51 * 0.1) = 5
3. 随机采样 5 个序列：[1, 7, 17, 40, 47]
4. 只处理采样到的序列
5. 正常生成数据集划分和标注
```

### 为什么按序列采样？

- **保持时序连续性**：一个视频序列内的帧是连续的，按序列采样可以保持这种连续性
- **验证跨场景泛化**：采样不同序列可以验证模型在不同场景下的表现
- **更快的处理速度**：跳过大部分序列处理，大幅提升速度
- **完整的数据集结构**：仍然是完整的三元划分（train/val/test）

### 可重现性

```python
def _sample_sequences(self, sequences):
    random.seed(42)  # 固定种子
    sampled = random.sample(sequences, n_sample)
    return sorted(sampled)
```

- 每次运行 simple 模式，采样结果都相同
- 便于调试和复现问题
- 最终提交时使用完整数据集

## 性能对比

### 处理时间（RTX 3090）

| 模式 | 序列数量 | 预估时间 | 生成图像数 |
|------|---------|---------|-----------|
| 完整模式 | 51 | ~1 小时 | 54,185 |
| simple (10%) | 5 | ~6 分钟 | ~5,300 |
| simple (5%) | 2-3 | ~3 分钟 | ~2,100 |
| simple (1%) | 1 | ~1 分钟 | ~1,000 |

### 存储占用

| 模式 | 输出目录大小 |
|------|------------|
| 完整 | ~200 MB (标注) |
| simple (10%) | ~20 MB (标注) |

## 应用场景

### ✅ 适合使用 simple 模式的场景

- **代码调试**：验证数据集生成逻辑是否正确
- **超参数调优**：快速测试不同的时间容差阈值
- **框架验证**：验证 COCO 格式是否正确
- **随机种子测试**：验证划分的一致性
- **快速演示**：向他人展示项目功能

### ❌ 不适合使用 simple 模式的场景

- **最终模型训练**
- **基准测试**
- **论文实验**
- **生产部署**

## 输出示例

### 命令执行

```bash
$ python convert_fred_to_fusion_v2.py --split-mode frame --simple --simple-ratio 0.1
```

### 输出日志

```
======================================================================
FRED 融合转换 - FRAME 级别划分
======================================================================
⚠️  简化模式已启用！采样比例: 10.0%
  This will use only 10.0% of the data for quick testing

找到 51 个序列: [0, 1, 2, ..., 50]
简化模式: 从 51 个序列中采样 5 个
采样序列: [1, 7, 17, 40, 47]

生成帧级别划分...
处理序列: 100%|██████████| 5/5 [00:45<00:00,  9.00s/it]

帧级别划分结果:
  TRAIN: 2,847 帧
  VAL: 1,210 帧
  TEST: 1,243 帧

处理 TRAIN 划分...
✓ 已保存: datasets/fred_fusion_v2/annotations/instances_train.json
  图像: 2,847, 标注: 2,847

处理 VAL 划分...
✓ 已保存: datasets/fred_fusion_v2/annotations/instances_val.json
  图像: 1,210, 标注: 1,210

处理 TEST 划分...
✓ 已保存: datasets/fred_fusion_v2/annotations/instances_test.json
  图像: 1,243, 标注: 1,243

=====================================================================
融合数据集生成完成！
=====================================================================
```

### 输出目录结构

```
datasets/fred_fusion_v2/
├── annotations/
│   ├── instances_train.json      # 2,847 张图像
│   ├── instances_val.json         # 1,210 张图像
│   └── instances_test.json        # 1,243 张图像
└── fusion_info_v2.json           # 融合信息
```

## 常见问题

### Q1: simple 模式采样的序列是固定的吗？

**A**: 是的，使用固定随机种子 42，每次运行相同参数都会采样相同的序列。

### Q2: 采样后还能保证 train/val/test 的比例吗？

**A**: 是的，采样是在序列级别进行的，然后在采样后的序列上进行正常的三元划分，保证比例关系。

### Q3: 采样比例越小，性能提升越明显吗？

**A**: 是的，近似线性关系。采样 10% 约提升 10 倍速度。

### Q4: simple 模式生成的数据集能用于训练吗？

**A**: 不推荐。simple 模式主要用于测试和调试，最终训练应使用完整数据集。

### Q5: 如何修改采样策略？

**A**: 编辑 `_sample_sequences` 方法，可以实现：
- 按帧采样
- 按特定条件采样
- 分层采样

## 性能优化建议

### 加速数据集生成

1. **使用 simple 模式调试**：先用 `--simple-ratio 0.05` 快速验证
2. **调整帧级别划分粒度**：
   ```python
   # 在 validator.py 中调整
   VAL_KEY_STEP = 16  # 增大步长减少验证帧数
   ```
3. **禁用验证**：`--validate-only` 仅生成不验证
4. **使用软链接**：
   ```python
   FREDFusionConverterV2(..., use_symlinks=True)  # 减少磁盘 IO
   ```

## 代码结构

### 主要修改

1. **构造函数参数**
   ```python
   def __init__(self, ..., simple=False, simple_ratio=0.1, ...):
   ```

2. **序列采样方法**
   ```python
   def _sample_sequences(self, sequences):
       random.seed(42)
       n_sample = max(1, int(len(sequences) * self.simple_ratio))
       return random.sample(sequences, n_sample)
   ```

3. **采样集成**
   ```python
   def generate_all_fusion(self, ...):
       sequences = self.get_all_sequences()
       
       if self.simple:
           sequences = self._sample_sequences(sequences)
           logger.info(f"简化模式: 采样 {len(sequences)} 个序列")
   ```

## 相关文档

- [FUSION_TRAINING_GUIDE.md](FUSION_TRAINING_GUIDE.md) - 融合数据集训练指南
- [FUSION_DATALOADER_GUIDE.md](FUSION_DATALOADER_GUIDE.md) - 数据加载器文档
- [FRAME_SPLIT_FIX_SUMMARY.md](FRAME_SPLIT_FIX_SUMMARY.md) - 帧级别划分修复记录
- [FUSION_TEST_REPORT.md](FUSION_TEST_REPORT.md) - 融合测试报告

## 总结

simple 模式是一个实用的快速测试工具：

✅ **大幅提升开发效率**：从 1 小时缩短到 5 分钟  
✅ **保持数据集完整性**：采样后仍保持完整的三元划分  
✅ **可重现的结果**：固定随机种子确保一致性  
✅ **灵活的参数调节**：可调整采样比例

赶快试试 `--simple` 参数，体验快速开发的乐趣吧！🚀