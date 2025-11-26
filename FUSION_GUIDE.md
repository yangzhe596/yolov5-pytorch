# FRED 多模态融合数据集 - 使用文档

## 概述

`convert_fred_to_fusion.py` 是一个专门用于生成 FRED 数据集多模态融合版本的脚本。它能够将 RGB 和 Event 数据根据精确的时间戳进行配对，生成支持双模态训练的 COCO 格式数据集。

## 核心特性

### 1. 时间戳精确配对
- **配对算法**: 以 RGB 帧为基准，查找最近的 Event 帧
- **容差阈值**: 默认 33ms（±16.5ms），可自定义
- **配对状态**: 
  - `dual`: 成功配对（时间差 ≤ 阈值）
  - `rgb_only`: 仅 RGB 帧（无匹配 Event）
  - `event_only`: 仅 Event 帧（无匹配 RGB）
  - `both_failed`: 都未匹配（理论上不会出现）

### 2. 融合标注格式
生成的 COCO 文件包含以下扩展字段：

**图像条目 (images)**:
```json
{
  "id": 1,
  "rgb_file_name": "0/RGB/Video_0_16_03_03.363444.jpg",
  "event_file_name": "0/Event/Frames/Video_0_frame_100032333.png",
  "width": 1280,
  "height": 720,
  "sequence_id": 0,
  "rgb_timestamp": 57783.363444,
  "event_timestamp": 57783.363440,
  "time_diff": 4e-06,        // 时间差（秒）
  "fusion_status": "dual",   // 融合状态
  "modality": "dual"         // 模态类型
}
```

**标注条目 (annotations)**:
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "area": 1711.44,
  "iscrowd": 0,
  "drone_id": 1,
  "modality": "dual"         // 标注所属模态
}
```

## 使用方法

### 1. 基础用法（推荐）

```bash
# 使用默认参数（33ms 容差，帧级别划分）
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py
```

### 2. 常用参数

```bash
# 自定义时间容差（20ms）
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py --threshold 0.02

# 序列级别划分（更好的泛化评估）
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py --split-mode sequence

# 简化模式（快速测试）
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py --simple

# 自定义数据集路径
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py \
    --fred-root /path/to/fred \
    --output-root /path/to/output
```

### 3. 完整参数列表

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--fred-root` | FRED 数据集根目录 | `/mnt/data/datasets/fred` | - |
| `--output-root` | 输出目录 | `datasets/fred_fusion` | - |
| `--split-mode` | 划分模式 | `frame` | `frame`/`sequence` |
| `--threshold` | 时间容差阈值（秒） | `0.033` | > 0 |
| `--train-ratio` | 训练集比例 | `0.7` | 0-1 |
| `--val-ratio` | 验证集比例 | `0.15` | 0-1 |
| `--test-ratio` | 测试集比例 | `0.15` | 0-1 |
| `--seed` | 随机种子 | `42` | 任意整数 |
| `--simple` | 启用简化模式 | `False` | `True`/`False` |
| `--simple-ratio` | 简化模式采样比例 | `0.1` | 0-1 |
| `--use-symlinks` | 使用软链接（不拷贝图片） | `False` | `True`/`False` |

## 时间容差阈值选择

### 推荐配置

| 场景 | 阈值 | 说明 |
|------|------|------|
| **严格配对** | 20ms | 要求极高时间同步，适合高频融合 |
| **标准配对** | 33ms | 接近 30FPS 帧率，推荐使用 |
| **宽松配对** | 50ms | 允许更大延迟，提高配对成功率 |
| **宽松配对** | 100ms | 宽松配对，适合长延迟场景 |

### 指导原则

1. **33ms (30FPS)**: 大多数场景的最佳选择，平衡配对数和精度
2. **20ms (50FPS)**: 高精度需求，但会丢失较多配对帧
3. **50ms+**: 适合 Event 数据较稀疏的场景

## 输出目录结构

```
datasets/fred_fusion/
├── annotations/
│   ├── instances_train.json      # 训练集标注
│   ├── instances_val.json        # 验证集标注
│   └── instances_test.json       # 测试集标注
├── fusion_info.json              # 融合配置信息
└── rgb_images/                   # RGB 图片目录（软链接）
└── event_images/                 # Event 图片目录（软链接）
```

## 数据统计示例

```bash
# 运行转换
/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py --simple

# 输出示例
==========================================
FRED 融合转换 - FRAME 级别划分
==========================================

序列 0: 100 RGB 帧, 150 Event 帧
处理配对帧 (序列 6): 100it [00:01, 85.32it/s, dual=85, rgb=15, event=0, matched=68]

==========================================
TRAIN 划分统计:
==========================================
  图像总数: 68
  标注总数: 68
  序列数: 42

  配对状态:
    - 双模态 (DUAL): 85 (85.0%)
    - 仅 RGB (RGB_ONLY): 15 (15.0%)
    - 仅 Event (EVENT_ONLY): 0 (0.0%)

  带标注的帧:
    - 双模态匹配: 58
    - RGB 匹配: 10
    - Event 匹配: 0
```

## 输出 JSON 格式验证

使用验证工具检查生成的数据集：

```bash
# 运行验证
/home/yz/.conda/envs/torch/bin/python3 verify_fusion_dataset.py

# 输出示例
========================================
验证: instances_train.json
========================================
✅ 基本结构验证通过
   - 图像数: 1478
   - 标注数: 1478
   - 类别数: 1

📊 融合状态分布:
   - DUAL: 1256/1478 (85.0%)
   - RGB_ONLY: 222/1478 (15.0%)
   - EVENT_ONLY: 0/1478 (0.0%)

⏱️  配对时间差统计 (单位: 秒):
   - 平均: 0.0043s (4.3ms)
   - 最大: 0.0328s (32.8ms)
   - 阈值: 33ms
   - 匹配率: 85.0%
```

## 常见问题

### Q1: 为什么部分 RGB 帧没有配对 Event 帧？

**原因**: 
- Event 帧较稀疏，某些时刻没有采集
- 时间差超过阈值（默认 33ms）

**解决**: 
- 放宽时间阈值：`--threshold 0.05`（50ms）
- 接受 `rgb_only` 状态，模型仍可训练

### Q2: 如何查看配对质量？

**方法**:
1. 查看转换日志中的配对统计
2. 使用 `verify_fusion_dataset.py` 分析
3. 检查 `fusion_info.json` 中的配置

### Q3: Fusion 模式如何在训练中使用？

**准备**: 
1. 确保 Fusion 数据集已生成
2. 创建新的训练脚本：`train_fusion.py`（需实现）
3. 修改模型输入层支持双模态

## 与单模态数据集对比

| 特性 | 原始 v2 脚本 | 融合脚本 |
|------|-------------|----------|
| 处理模态 | RGB 或 Event | RGB + Event |
| 时间配对 | 不支持 | 支持（±33ms） |
| 输入字段 | 单个 file_name | rgb_file_name + event_file_name |
| 标注模态 | 无 | 标注所属模态 |
| 输出目录 | fred_coco/rgb, fred_coco/event | fred_fusion/ |
| 兼容性 | YOLOv5 单模态 | 需要扩展支持双模态 |

## 快速测试

```bash
# 使用简化模式快速测试
bash quick_test_fusion.sh --simple

# 自定义参数测试
bash quick_test_fusion.sh --threshold 0.02 --mode sequence

# 查看帮助
bash quick_test_fusion.sh --help
```

## 下一步工作

1. **实现融合训练脚本**: `train_fusion.py`
2. **添加双模态可视化**: `visualize_fusion.py`
3. **扩展 YOLOv5 模型**: 支持双通道输入
4. **性能评估**: 对比单模态 vs 融合效果

---

**最后更新**: 2025-11-24  
**作者**: MiAIA Group  
**许可证**: Apache License 2.0