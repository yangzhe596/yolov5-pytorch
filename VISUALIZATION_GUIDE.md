# FRED 数据集可视化指南

本文档说明如何使用 `visualize_fred_sequences.py` 脚本可视化 FRED 数据集的序列和标注。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [使用示例](#使用示例)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 功能特性

### 核心功能

- ✅ **序列可视化**: 可视化整个视频序列的标注
- ✅ **视频导出**: 导出为 MP4 视频文件
- ✅ **双模态支持**: 支持 RGB 和 Event 两种模态
- ✅ **对比视频**: 创建 RGB 和 Event 并排对比视频
- ✅ **实时显示**: 实时显示标注信息
- ✅ **多序列处理**: 批量处理多个序列
- ✅ **快速预览**: 支持限制帧数快速预览

### 显示信息

- 边界框（不同 drone_id 使用不同颜色）
- Drone ID 标签
- 序列信息（序列号、模态）
- 帧信息（当前帧/总帧数）
- 时间戳
- 目标数量

---

## 快速开始

### 步骤 1: 确保数据集已准备好

```bash
# 检查 COCO 格式数据集
ls datasets/fred_coco/rgb/annotations/
ls datasets/fred_coco/event/annotations/
```

### 步骤 2: 列出可用序列

```bash
python visualize_fred_sequences.py --modality rgb --list-sequences
```

### 步骤 3: 可视化单个序列

```bash
# 仅显示窗口（不导出视频）
python visualize_fred_sequences.py --modality rgb --sequence 0

# 导出为视频
python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video
```

### 步骤 4: 查看导出的视频

```bash
# 视频保存在 visualizations/ 目录
ls visualizations/
```

---

## 详细使用

### 命令行参数

```bash
python visualize_fred_sequences.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fred-root` | str | `/mnt/data/datasets/fred` | FRED 数据集根目录 |
| `--coco-root` | str | `datasets/fred_coco` | COCO 格式数据集根目录 |
| `--modality` | str | `rgb` | 模态选择: `rgb` 或 `event` |
| `--sequence` | int | - | 单个序列 ID |
| `--sequences` | int+ | - | 多个序列 ID（空格分隔） |
| `--random` | flag | - | 随机选择一个序列 |
| `--export-video` | flag | - | 导出为视频文件 |
| `--output-dir` | str | `visualizations` | 输出目录 |
| `--fps` | int | `30` | 视频帧率 |
| `--no-window` | flag | - | 不显示窗口（仅导出视频） |
| `--max-frames` | int | - | 最大帧数（快速预览） |
| `--list-sequences` | flag | - | 列出所有可用序列 |
| `--comparison` | flag | - | 创建 RGB 和 Event 对比视频 |

---

## 使用示例

### 示例 1: 列出所有可用序列

```bash
python visualize_fred_sequences.py --modality rgb --list-sequences
```

**输出:**
```
可用序列 (50 个):
  序列 0: 3233 帧, 3850 标注, 时长 107.6s, Drones: [1]
  序列 1: 2474 帧, 2948 标注, 时长 82.4s, Drones: [1]
  序列 3: 2867 帧, 3440 标注, 时长 95.5s, Drones: [1]
  ...
```

### 示例 2: 可视化单个序列（仅显示窗口）

```bash
python visualize_fred_sequences.py --modality rgb --sequence 0
```

**效果:**
- 打开窗口显示序列
- 实时显示标注
- 按 `q` 退出
- 按 `p` 暂停

### 示例 3: 导出为视频

```bash
python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video
```

**输出:**
```
导出视频: visualizations/sequence_0_rgb.mp4
分辨率: 1280x720, 帧率: 30 FPS
处理序列 0: 100%|████████████| 3233/3233 [01:25<00:00, 37.85it/s]
✓ 视频已保存: visualizations/sequence_0_rgb.mp4
```

### 示例 4: 可视化多个序列

```bash
python visualize_fred_sequences.py --modality rgb --sequences 0 1 5 --export-video
```

**效果:**
- 依次处理序列 0, 1, 5
- 每个序列导出一个视频
- 显示总体统计信息

### 示例 5: 随机序列

```bash
python visualize_fred_sequences.py --modality event --random --export-video
```

**效果:**
- 随机选择一个序列
- 导出为视频

### 示例 6: 快速预览（前 100 帧）

```bash
python visualize_fred_sequences.py --modality rgb --sequence 0 --max-frames 100
```

**效果:**
- 仅处理前 100 帧
- 快速查看序列内容

### 示例 7: 仅导出视频（不显示窗口）

```bash
python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video --no-window
```

**效果:**
- 不打开显示窗口
- 仅导出视频文件
- 适合批量处理

### 示例 8: 创建 RGB 和 Event 对比视频

```bash
python visualize_fred_sequences.py --comparison --sequence 0
```

**效果:**
- 创建并排对比视频
- 左侧: RGB 模态
- 右侧: Event 模态
- 输出: `visualizations/sequence_0_comparison.mp4`

### 示例 9: 自定义输出目录和帧率

```bash
python visualize_fred_sequences.py \
    --modality rgb \
    --sequence 0 \
    --export-video \
    --output-dir my_visualizations \
    --fps 60
```

**效果:**
- 输出到 `my_visualizations/` 目录
- 使用 60 FPS 帧率

### 示例 10: 批量处理所有序列

```bash
# 获取所有序列 ID
python visualize_fred_sequences.py --modality rgb --list-sequences | \
    grep "序列" | awk '{print $2}' | tr -d ':' > sequences.txt

# 批量处理（使用脚本）
for seq in $(cat sequences.txt); do
    python visualize_fred_sequences.py \
        --modality rgb \
        --sequence $seq \
        --export-video \
        --no-window
done
```

---

## 输出说明

### 视频文件

**命名格式:**
- 单模态: `sequence_{序列ID}_{模态}.mp4`
  - 例: `sequence_0_rgb.mp4`, `sequence_1_event.mp4`
- 对比视频: `sequence_{序列ID}_comparison.mp4`
  - 例: `sequence_0_comparison.mp4`

**视频属性:**
- 编码: MP4V
- 分辨率: 1280x720（单模态）或 2560x720（对比视频）
- 帧率: 默认 30 FPS（可自定义）

### 统计信息

每个序列处理完成后会显示统计信息：

```
======================================================================
序列 0 统计信息
======================================================================
总帧数: 3233
有标注的帧: 3233 (100.0%)
总标注数: 3850
平均标注/帧: 1.19
Drone IDs: [1]
======================================================================
```

### 总体统计

处理多个序列后会显示总体统计：

```
======================================================================
总体统计 (3 个序列)
======================================================================
总帧数: 8574
总标注数: 10238
平均标注/帧: 1.19
所有 Drone IDs: [1, 2]
======================================================================
```

---

## 可视化效果

### 边界框颜色

不同的 drone_id 使用不同的颜色：

| Drone ID | 颜色 | BGR 值 |
|----------|------|--------|
| 1 | 绿色 | (0, 255, 0) |
| 2 | 蓝色 | (255, 0, 0) |
| 3 | 红色 | (0, 0, 255) |
| 4 | 青色 | (255, 255, 0) |
| 5 | 品红色 | (255, 0, 255) |

### 信息面板

左上角显示信息面板：
```
┌─────────────────────────────────┐
│ Sequence: 0 (RGB)               │
│ Frame: 1234/3233                │
│ Time: 41.13s                    │
│ Objects: 1                      │
└─────────────────────────────────┘
```

### 标签

每个边界框上方显示标签：
```
┌──────────┐
│ Drone 1  │
└──────────┘
```

---

## 交互控制

### 窗口显示模式

在窗口显示模式下，支持以下按键：

| 按键 | 功能 |
|------|------|
| `q` | 退出可视化 |
| `p` | 暂停/继续 |
| 任意键 | 继续（暂停时） |

---

## 性能优化

### 处理速度

- **RGB 模态**: 约 30-40 帧/秒
- **Event 模态**: 约 30-40 帧/秒
- **对比视频**: 约 20-30 帧/秒

### 内存使用

- 单序列: 约 500 MB
- 多序列: 每个序列约 500 MB

### 加速技巧

1. **使用 `--no-window`**: 不显示窗口可提升 20-30% 速度
2. **使用 `--max-frames`**: 快速预览时限制帧数
3. **批量处理**: 使用脚本批量处理多个序列

---

## 常见问题

### Q1: 为什么有些序列没有标注？

**A:** 序列 2 在原始数据集中缺少标注文件，这是正常的。使用 `--list-sequences` 可以查看所有有标注的序列。

### Q2: 如何调整视频质量？

**A:** 当前使用 MP4V 编码。如需更高质量，可以修改脚本中的 `fourcc` 参数：

```python
# 高质量 H.264 编码
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```

### Q3: 视频播放速度太快/太慢？

**A:** 调整 `--fps` 参数：

```bash
# 慢速播放
python visualize_fred_sequences.py --sequence 0 --export-video --fps 15

# 快速播放
python visualize_fred_sequences.py --sequence 0 --export-video --fps 60
```

### Q4: 如何只可视化特定时间段？

**A:** 使用 `--max-frames` 参数：

```bash
# 仅前 300 帧（约 10 秒 @ 30 FPS）
python visualize_fred_sequences.py --sequence 0 --max-frames 300
```

### Q5: 窗口显示太小/太大？

**A:** OpenCV 窗口大小由图像分辨率决定（1280x720）。可以手动调整窗口大小。

### Q6: 如何批量导出所有序列？

**A:** 使用 Bash 脚本：

```bash
#!/bin/bash
# export_all_sequences.sh

MODALITY="rgb"
OUTPUT_DIR="all_visualizations"

# 获取所有序列
SEQUENCES=$(python visualize_fred_sequences.py --modality $MODALITY --list-sequences | \
            grep "序列" | awk '{print $2}' | tr -d ':')

# 批量处理
for seq in $SEQUENCES; do
    echo "处理序列 $seq..."
    python visualize_fred_sequences.py \
        --modality $MODALITY \
        --sequence $seq \
        --export-video \
        --no-window \
        --output-dir $OUTPUT_DIR
done

echo "完成！所有视频已保存到 $OUTPUT_DIR/"
```

### Q7: 对比视频中两个模态不同步？

**A:** 脚本使用较短序列的帧数，并按索引对齐。如果两个模态的帧数差异很大，可能会出现不同步。这是正常的，因为 Event 相机的采样率略高于 RGB。

### Q8: 如何修改边界框颜色？

**A:** 编辑脚本中的 `colors` 字典：

```python
self.colors = {
    1: (0, 255, 0),    # drone_id 1: 绿色
    2: (255, 0, 0),    # drone_id 2: 蓝色
    # 添加更多颜色...
}
```

### Q9: 视频文件太大？

**A:** 
- 使用更低的帧率: `--fps 15`
- 使用 `--max-frames` 限制帧数
- 使用更高效的编码（需修改脚本）

### Q10: 如何在视频中添加更多信息？

**A:** 编辑脚本中的 `info_text` 列表：

```python
info_text = [
    f"Sequence: {sequence_id} ({self.modality.upper()})",
    f"Frame: {idx + 1}/{len(images)}",
    f"Time: {img_info['timestamp']:.2f}s",
    f"Objects: {len(annotations)}",
    # 添加更多信息...
    f"FPS: {fps}",
    f"Resolution: {width}x{height}"
]
```

---

## 高级用法

### 创建自定义可视化脚本

```python
from visualize_fred_sequences import FREDSequenceVisualizer

# 创建可视化器
visualizer = FREDSequenceVisualizer(
    fred_root='/mnt/data/datasets/fred',
    coco_root='datasets/fred_coco',
    modality='rgb'
)

# 获取序列信息
info = visualizer.get_sequence_info(0)
print(f"序列 0: {info['n_images']} 帧")

# 可视化序列
stats = visualizer.visualize_sequence(
    sequence_id=0,
    export_video=True,
    output_dir='my_output',
    fps=30,
    show_window=False
)

print(f"处理了 {stats['total_frames']} 帧")
```

### 批量创建对比视频

```bash
#!/bin/bash
# create_all_comparisons.sh

for seq in 0 1 3 4 5; do
    python visualize_fred_sequences.py \
        --comparison \
        --sequence $seq \
        --output-dir comparisons \
        --fps 30
done
```

---

## 输出示例

### 单序列可视化

```bash
$ python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video

======================================================================
可视化序列 0 (RGB 模态)
======================================================================
总帧数: 3233
导出视频: visualizations/sequence_0_rgb.mp4
分辨率: 1280x720, 帧率: 30 FPS
处理序列 0: 100%|████████████████| 3233/3233 [01:25<00:00, 37.85it/s]
✓ 视频已保存: visualizations/sequence_0_rgb.mp4

======================================================================
序列 0 统计信息
======================================================================
总帧数: 3233
有标注的帧: 3233 (100.0%)
总标注数: 3850
平均标注/帧: 1.19
Drone IDs: [1]
======================================================================

✅ 可视化完成！
```

### 多序列可视化

```bash
$ python visualize_fred_sequences.py --modality rgb --sequences 0 1 5 --export-video

[处理序列 0...]
[处理序列 1...]
[处理序列 5...]

======================================================================
总体统计 (3 个序列)
======================================================================
总帧数: 8574
总标注数: 10238
平均标注/帧: 1.19
所有 Drone IDs: [1]
======================================================================

✅ 可视化完成！
```

---

## 相关文档

- `FRED_DATASET_CONVERSION_GUIDE.md` - 数据集转换指南
- `DATASET_GENERATION_IMPROVEMENTS.md` - 数据集生成改进
- `AGENTS.md` - 完整项目文档

---

**最后更新**: 2025-11-01  
**版本**: 1.0  
**维护者**: YOLOv5-PyTorch 项目组
