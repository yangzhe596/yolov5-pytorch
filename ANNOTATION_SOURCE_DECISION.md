# RGB数据标注来源决策

## 问题总结

FRED数据集的RGB模态存在**两个完全不同的标注来源**：

### 标注源对比（以Video_3_16_46_03.278530.jpg为例）

| 标注源 | 边界框位置 | 尺寸 | 位置描述 |
|--------|-----------|------|---------|
| **RGB_YOLO/** | xmin=847.9, ymin=313.2<br>xmax=904.9, ymax=351.8 | 57×38 px | 图片右上部 |
| **coordinates.txt** | xmin=252.0, ymin=665.0<br>xmax=371.0, ymax=720.0 | 119×55 px | 图片左下部 |
| **中心点距离** | **669.9 像素** | - | **完全不同的目标！** |

## 可视化结果

已生成可视化对比图：`annotation_comparison_3_Video_3_16_46_03.278530.png`

请查看该图片，你会看到：
- **红色框**（RGB_YOLO）: 在图片右上部
- **蓝色框**（coordinates.txt）: 在图片左下部

## 关键问题 ❓

**请确认以下问题以决定使用哪个标注源**：

### 问题1: 标注的目标是什么？
- [ ] 无人机/飞行器
- [ ] 鸟类
- [ ] 车辆
- [ ] 人
- [ ] 其他: ___________

### 问题2: 哪个标注源是正确的？

查看可视化图片后，请确认：
- [ ] **RGB_YOLO/** 标注的目标是正确的（红色框）
- [ ] **coordinates.txt** 标注的目标是正确的（蓝色框）
- [ ] 两者都对，但标注的是不同的目标
- [ ] 不确定，需要查看更多样本

### 问题3: coordinates.txt的用途？
- [ ] 原始的人工标注
- [ ] 自动跟踪算法的输出
- [ ] 某个特定目标的轨迹
- [ ] 已废弃的旧标注
- [ ] 其他: ___________

## 当前状态

**当前COCO转换使用**: RGB_YOLO/ 目录

**转换结果**:
- ✅ RGB模态: 19,471张图片
- ✅ Event模态: 28,714张图片
- ✅ 格式正确，可以使用

## 可能的解决方案

### 方案A: 使用RGB_YOLO（当前方案）

**适用情况**: 如果RGB_YOLO标注的目标是正确的

**优点**:
- ✅ 已经完成转换
- ✅ 格式标准
- ✅ 可以直接使用

**行动**: 无需修改，继续使用当前的COCO数据集

### 方案B: 使用coordinates.txt

**适用情况**: 如果coordinates.txt标注的目标是正确的

**需要做的**:
1. 创建新的转换脚本
2. 建立图片文件名和coordinates.txt时间戳的映射关系
3. 重新生成COCO数据集

**挑战**:
- ⚠️ 需要确定时间戳的对应关系
- ⚠️ coordinates.txt的时间戳是相对时间，需要找到起始点
- ⚠️ 可能不是所有图片都有对应的标注

### 方案C: 使用两个标注源（多类别）

**适用情况**: 如果两个标注源都对，但标注的是不同类别的目标

**需要做的**:
1. 定义两个类别（如：class 0 = RGB_YOLO的目标，class 1 = coordinates.txt的目标）
2. 合并两个标注源
3. 重新生成COCO数据集

## 验证步骤

### 步骤1: 查看可视化图片

```bash
# 已生成的图片
ls annotation_comparison_*.png
```

### 步骤2: 检查更多样本

```bash
# 可视化其他样本
/home/yz/miniforge3/envs/torch/bin/python3 visualize_annotation_sources.py \
    --video_id 3 --image_name Video_3_16_46_10.014396.jpg

/home/yz/miniforge3/envs/torch/bin/python3 visualize_annotation_sources.py \
    --video_id 0 --image_name Video_0_16_03_49.204702.jpg
```

### 步骤3: 检查数据集文档

查看FRED数据集是否有README或文档说明标注格式。

## 建议

在确认正确的标注源之前：

1. ✅ **保留当前的COCO数据集**（基于RGB_YOLO）
2. 📋 **查看可视化图片**，确认哪个标注是正确的
3. 📋 **检查多个样本**，确保结论的一致性
4. 📋 **查找数据集文档**，了解标注的含义
5. 📋 **根据确认结果决定**是否需要重新转换

## 快速测试命令

```bash
# 可视化对比（需要确认哪个标注正确）
/home/yz/miniforge3/envs/torch/bin/python3 visualize_annotation_sources.py \
    --video_id 3 --image_name Video_3_16_46_03.278530.jpg

# 查看生成的对比图
ls -lh annotation_comparison_*.png

# 检查其他样本
/home/yz/miniforge3/envs/torch/bin/python3 visualize_annotation_sources.py \
    --video_id 3 --image_name Video_3_16_46_05.012063.jpg
```

## 下一步

**请提供反馈**：

1. 查看生成的可视化图片 `annotation_comparison_3_Video_3_16_46_03.278530.png`
2. 告诉我哪个标注框（红色或蓝色）标注的目标是正确的
3. 如果需要，我将基于正确的标注源重新生成COCO数据集

---

**创建时间**: 2025-10-20  
**状态**: 等待确认正确的标注源  
**可视化图片**: annotation_comparison_3_Video_3_16_46_03.278530.png
