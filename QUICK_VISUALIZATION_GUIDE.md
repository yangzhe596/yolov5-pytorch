# FRED 数据集可视化 - 快速指南

## 🎬 一键生成可视化视频

### 方法 1: 使用快捷脚本（最简单）

```bash
# 快速预览（前100帧）
./quick_visualize.sh 0 rgb 100

# 完整序列
./quick_visualize.sh 0 rgb

# Event 模态
./quick_visualize.sh 0 event 100
```

### 方法 2: 使用 Python 脚本

```bash
# 快速预览（50帧，约2秒）
python visualize_fred_sequences.py \
    --modality rgb \
    --sequence 0 \
    --export-video \
    --no-window \
    --max-frames 50

# 完整序列（1316帧，约15秒）
python visualize_fred_sequences.py \
    --modality rgb \
    --sequence 21 \
    --export-video \
    --no-window

# RGB 和 Event 对比视频
python visualize_fred_sequences.py \
    --comparison \
    --sequence 0
```

---

## ⚡ 性能表现

### 实测性能（RTX 3090）

| 序列 | 帧数 | 处理时间 | 速度 | 视频大小 |
|------|------|---------|------|---------|
| 序列 0（预览） | 50 | 1.3秒 | 38 FPS | 1.2 MB |
| 序列 21（完整） | 1316 | 14.9秒 | 93 FPS | 9.9 MB |
| 序列 0（完整） | 1909 | ~20秒 | 95 FPS | ~15 MB |

### 性能优化亮点

- ✅ 标注加载速度提升 **60倍**（0.5秒 vs 30秒）
- ✅ 帧处理速度达到 **90-100 FPS**
- ✅ 1000帧序列仅需 **10-12秒**
- ✅ 内存占用减少 **37.5%**

---

## 📊 生成的视频示例

### 视频文件位置

```
visualizations/
├── sequence_0_rgb.mp4          # 序列0 RGB模态
├── sequence_0_event.mp4        # 序列0 Event模态
├── sequence_0_comparison.mp4   # 序列0 对比视频
├── sequence_21_rgb.mp4         # 序列21 RGB模态
└── ...
```

### 视频内容

每个视频包含：
- ✅ 边界框（绿色/蓝色/红色，根据 drone_id）
- ✅ Drone ID 标签（D1, D2, ...）
- ✅ 序列信息（序列号、模态）
- ✅ 帧信息（当前帧/总帧数）
- ✅ 时间戳
- ✅ 目标数量

---

## 🎯 推荐工作流程

### 1. 快速检查数据质量

```bash
# 随机选择3个序列，每个预览100帧
for i in 0 5 10; do
    ./quick_visualize.sh $i rgb 100
done
```

**耗时：** 约 10 秒
**目的：** 快速验证数据集质量

### 2. 生成完整序列视频

```bash
# 选择几个代表性序列
./quick_visualize.sh 0 rgb      # 序列0（完整）
./quick_visualize.sh 21 event   # 序列21（Event模态）
```

**耗时：** 每个序列 15-30 秒
**目的：** 详细检查标注准确性

### 3. 创建对比视频

```bash
# RGB vs Event 对比
python visualize_fred_sequences.py --comparison --sequence 0
```

**耗时：** 约 30-40 秒
**目的：** 对比两种模态的标注一致性

---

## 📝 常用命令速查

### 列出所有序列

```bash
python visualize_fred_sequences.py --modality rgb --list-sequences
```

### 快速预览

```bash
# 前50帧（约2秒）
./quick_visualize.sh 0 rgb 50

# 前100帧（约3秒）
./quick_visualize.sh 0 rgb 100
```

### 完整导出

```bash
# RGB 模态
./quick_visualize.sh 0 rgb

# Event 模态
./quick_visualize.sh 0 event
```

### 批量处理

```bash
# 批量导出序列 0-5
for i in {0..5}; do
    ./quick_visualize.sh $i rgb
done
```

### 对比视频

```bash
python visualize_fred_sequences.py --comparison --sequence 0
```

---

## 🎨 视频效果说明

### 边界框颜色

| Drone ID | 颜色 | 用途 |
|----------|------|------|
| 1 | 绿色 | 主要无人机 |
| 2 | 蓝色 | 第二架无人机 |
| 3 | 红色 | 第三架无人机 |

### 信息显示

```
┌────────────────────────────────────────────────┐
│ Seq:0 Frame:1234/1909 Time:41.1s Obj:1        │
└────────────────────────────────────────────────┘
```

---

## ⏱️ 预期处理时间

### 不同帧数的处理时间

| 帧数 | 处理时间 | 视频大小 |
|------|---------|---------|
| 50 | ~2秒 | ~1 MB |
| 100 | ~3秒 | ~2 MB |
| 500 | ~6秒 | ~5 MB |
| 1000 | ~11秒 | ~10 MB |
| 2000 | ~22秒 | ~20 MB |
| 3000 | ~33秒 | ~30 MB |

### 不同序列的处理时间

| 序列 | 帧数 | 预期时间 |
|------|------|---------|
| 21 | 1316 | ~15秒 |
| 0 | 1909 | ~20秒 |
| 1 | 2379 | ~26秒 |
| 5 | 2984 | ~32秒 |

---

## 🔍 验证数据正确性

### 检查要点

观看生成的视频时，检查：

1. ✅ **边界框位置**: 是否准确框住无人机
2. ✅ **边界框大小**: 是否合理（不过大/过小）
3. ✅ **时间连续性**: 边界框是否平滑移动
4. ✅ **ID 一致性**: 同一无人机的 ID 是否保持一致
5. ✅ **遮挡处理**: 遮挡时标注是否合理
6. ✅ **边界情况**: 无人机进出画面时的标注

### 常见问题

**边界框抖动：**
- 正常现象（插值标注）
- 如果抖动严重，可能是标注问题

**ID 切换：**
- 可能是多个无人机
- 检查是否有 ID 2, 3 等

**边界框缺失：**
- 可能是时间戳不匹配
- 检查日志中的匹配率

---

## 📚 相关文档

- `VISUALIZATION_GUIDE.md` - 详细可视化指南
- `VISUALIZATION_PERFORMANCE.md` - 性能优化说明
- `FRED_DATASET_CONVERSION_GUIDE.md` - 数据集转换指南

---

## 🎉 快速开始示例

```bash
# 1. 列出所有序列
python visualize_fred_sequences.py --modality rgb --list-sequences

# 2. 快速预览序列0（前100帧）
./quick_visualize.sh 0 rgb 100

# 3. 查看生成的视频
ls -lh visualizations/

# 4. 播放视频（使用系统默认播放器）
xdg-open visualizations/sequence_0_rgb.mp4
# 或
vlc visualizations/sequence_0_rgb.mp4
```

---

**最后更新**: 2025-11-01  
**版本**: 2.0  
**处理速度**: 90-100 FPS  
**维护者**: YOLOv5-PyTorch 项目组
