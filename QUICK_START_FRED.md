# FRED数据集快速开始指南

## 概述

FRED数据集已成功转换为COCO格式，包含RGB和Event两种模态。本指南将帮助你快速开始使用这些数据。

## 环境要求

- Python环境: `/home/yz/miniforge3/envs/torch/bin/python3`
- 已安装的包: torch, PIL, matplotlib, tqdm

## 数据集位置

```
/mnt/data/code/yolov5-pytorch/datasets/fred_coco/
├── rgb/      # RGB模态 (19,471张图片)
└── event/    # Event模态 (28,714张图片)
```

## 快速命令

### 1. 查看数据集信息

```bash
# RGB训练集
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py \
    --modality rgb --split train

# Event验证集
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py \
    --modality event --split val

# 测试集
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py \
    --modality rgb --split test
```

### 2. 验证数据集

```bash
# 验证RGB训练集
/home/yz/miniforge3/envs/torch/bin/python3 verify_coco_dataset.py \
    --modality rgb --split train --show_samples 0

# 验证Event训练集
/home/yz/miniforge3/envs/torch/bin/python3 verify_coco_dataset.py \
    --modality event --split train --show_samples 0
```

### 3. 可视化样本（需要GUI环境）

```bash
# 可视化RGB样本
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py \
    --modality rgb --split train --visualize --num_samples 5

# 可视化Event样本
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py \
    --modality event --split train --visualize --num_samples 5
```

## Python代码示例

### 基本加载

```python
import json
from pathlib import Path

# 加载RGB训练集标注
with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    coco_data = json.load(f)

print(f"图片数: {len(coco_data['images'])}")
print(f"标注数: {len(coco_data['annotations'])}")
print(f"类别: {[c['name'] for c in coco_data['categories']]}")
```

### 使用PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

class FREDDataset(Dataset):
    def __init__(self, coco_root, split='train'):
        # 加载标注
        ann_file = f"{coco_root}/annotations/instances_{split}.json"
        with open(ann_file) as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.split = split
        self.coco_root = coco_root
        
        # 创建映射
        self.img_to_anns = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        # 加载图片
        img_path = f"{self.coco_root}/{self.split}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        
        # 获取标注
        anns = self.img_to_anns.get(img_info['id'], [])
        
        # 提取边界框
        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])  # 转为[x1,y1,x2,y2]
        
        return image, torch.tensor(boxes, dtype=torch.float32)

# 使用
dataset = FREDDataset('datasets/fred_coco/rgb', split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for images, boxes in loader:
    print(f"Batch size: {len(images)}")
    break
```

### 统计分析

```python
import json
import numpy as np

# 加载数据
with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    data = json.load(f)

# 边界框尺寸分析
widths = [ann['bbox'][2] for ann in data['annotations']]
heights = [ann['bbox'][3] for ann in data['annotations']]

print("边界框宽度统计:")
print(f"  均值: {np.mean(widths):.2f}")
print(f"  中位数: {np.median(widths):.2f}")
print(f"  标准差: {np.std(widths):.2f}")

print("\n边界框高度统计:")
print(f"  均值: {np.mean(heights):.2f}")
print(f"  中位数: {np.median(heights):.2f}")
print(f"  标准差: {np.std(heights):.2f}")

# 宽高比分析
aspect_ratios = [w/h for w, h in zip(widths, heights)]
print(f"\n宽高比: {np.mean(aspect_ratios):.2f} ± {np.std(aspect_ratios):.2f}")
```

## 数据集统计摘要

### RGB模态
- **总计**: 19,471 张图片
- **训练集**: 13,629 张 (70%)
- **验证集**: 3,894 张 (20%)
- **测试集**: 1,948 张 (10%)
- **图片格式**: JPG
- **图片尺寸**: 1280 x 720
- **平均边界框**: 50.22 x 34.08 像素

### Event模态
- **总计**: 28,714 张图片
- **训练集**: 20,099 张 (70%)
- **验证集**: 5,742 张 (20%)
- **测试集**: 2,873 张 (10%)
- **图片格式**: PNG
- **图片尺寸**: 1280 x 720
- **平均边界框**: 50.96 x 34.58 像素

## 常见任务

### 任务1: 查看特定图片

```python
import json
from PIL import Image
import matplotlib.pyplot as plt

# 加载标注
with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    data = json.load(f)

# 获取第一张图片
img_info = data['images'][0]
img_path = f"datasets/fred_coco/rgb/train/{img_info['file_name']}"

# 显示
img = Image.open(img_path)
plt.imshow(img)
plt.title(f"{img_info['file_name']}\n{img_info['width']}x{img_info['height']}")
plt.axis('off')
plt.show()
```

### 任务2: 统计每个视频序列的图片数

```python
import json
from collections import Counter

with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    data = json.load(f)

# 提取视频ID（文件名的第一个字符）
video_ids = [img['file_name'].split('_')[0] for img in data['images']]
video_counts = Counter(video_ids)

print("每个视频序列的图片数:")
for vid, count in sorted(video_counts.items()):
    print(f"  视频 {vid}: {count} 张")
```

### 任务3: 导出为其他格式

```python
import json

# 加载COCO数据
with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    coco_data = json.load(f)

# 转换为简单格式
simple_format = []
for img in coco_data['images']:
    img_id = img['id']
    # 找到对应的标注
    anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
    
    for ann in anns:
        simple_format.append({
            'image': img['file_name'],
            'bbox': ann['bbox'],
            'category': ann['category_id']
        })

# 保存
with open('simple_annotations.json', 'w') as f:
    json.dump(simple_format, f, indent=2)

print(f"已导出 {len(simple_format)} 条标注")
```

## 下一步

1. **探索数据**: 使用提供的脚本查看和分析数据
2. **数据增强**: 根据需要实现数据增强策略
3. **模型训练**: 集成到你的训练流程中
4. **性能评估**: 使用测试集评估模型性能

## 文件说明

- `convert_fred_to_coco.py` - 数据集转换脚本
- `verify_coco_dataset.py` - 数据集验证脚本
- `example_load_coco.py` - 使用示例脚本
- `README_FRED_COCO.md` - 详细文档
- `FRED_DATASET_SUMMARY.md` - 数据集摘要
- `QUICK_START_FRED.md` - 本文档

## 获取帮助

```bash
# 查看脚本帮助
/home/yz/miniforge3/envs/torch/bin/python3 convert_fred_to_coco.py --help
/home/yz/miniforge3/envs/torch/bin/python3 verify_coco_dataset.py --help
/home/yz/miniforge3/envs/torch/bin/python3 example_load_coco.py --help
```

## 故障排除

### 问题: 找不到图片文件
**解决**: 检查路径是否正确，确保数据集已完整转换

### 问题: 内存不足
**解决**: 减小batch_size或使用数据生成器

### 问题: 可视化无法显示
**解决**: 在无GUI环境中，图片会保存但不会显示，这是正常的

---

**创建日期**: 2025-10-20  
**Python环境**: /home/yz/miniforge3/envs/torch/bin/python3  
**项目路径**: /mnt/data/code/yolov5-pytorch
