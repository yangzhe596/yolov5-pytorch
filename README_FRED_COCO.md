# FRED数据集COCO格式转换指南

## 数据集概述

FRED数据集包含两种模态的数据：
- **RGB模态**: 标准RGB相机图像 (1280x720, JPG格式)
- **Event模态**: 事件相机图像 (1280x720, PNG格式)

每个模态都有对应的YOLO格式标注文件。

## 数据集统计

根据初步分析：

| 视频序列 | RGB图片数 | RGB有效标注 | Event图片数 | Event有效标注 |
|---------|----------|------------|------------|--------------|
| 0       | 3,233    | 1,302      | 3,348      | 1,917        |
| 1       | 3,399    | 1,651      | ~3,400     | 2,474        |
| 2       | 0        | 0          | 0          | 0            |
| 3       | 3,439    | 2,020      | ~3,440     | 2,867        |
| 4       | 3,460    | 2,086      | ~3,460     | 3,093        |
| 5       | 3,387    | 2,097      | ~3,390     | 3,036        |
| 6-10    | ~16,000  | ~10,000    | ~16,000    | ~15,000      |

**注**: 序列2为空，其他序列包含有效数据。

## 转换为COCO格式

### 1. 基本用法

```bash
# 转换RGB模态
/home/yz/miniforge3/envs/torch/bin/python3 convert_fred_to_coco.py --modality rgb

# 转换Event模态
/home/yz/miniforge3/envs/torch/bin/python3 convert_fred_to_coco.py --modality event

# 同时转换两种模态
/home/yz/miniforge3/envs/torch/bin/python3 convert_fred_to_coco.py --modality both
```

### 2. 自定义参数

```bash
/home/yz/miniforge3/envs/torch/bin/python3 convert_fred_to_coco.py \
    --modality rgb \
    --fred_root /mnt/data/datasets/fred \
    --output_root datasets/fred_coco \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --seed 42
```

### 3. 参数说明

- `--modality`: 选择模态 (rgb/event/both)
- `--fred_root`: FRED数据集根目录 (默认: /mnt/data/datasets/fred)
- `--output_root`: 输出目录 (默认: datasets/fred_coco)
- `--train_ratio`: 训练集比例 (默认: 0.7)
- `--val_ratio`: 验证集比例 (默认: 0.2)
- `--test_ratio`: 测试集比例 (默认: 0.1)
- `--seed`: 随机种子 (默认: 42)

## 输出目录结构

转换后的COCO格式数据集结构如下：

```
datasets/fred_coco/
├── rgb/                          # RGB模态
│   ├── annotations/
│   │   ├── instances_train.json  # 训练集标注
│   │   ├── instances_val.json    # 验证集标注
│   │   └── instances_test.json   # 测试集标注
│   ├── train/                    # 训练集图片
│   │   ├── 0_Video_0_16_03_49.204702.jpg
│   │   └── ...
│   ├── val/                      # 验证集图片
│   ├── test/                     # 测试集图片
│   └── dataset_info.txt          # 数据集信息
│
└── event/                        # Event模态
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    │   └── instances_test.json
    ├── train/
    │   ├── 0_Video_0_frame_48299517.png
    │   └── ...
    ├── val/
    ├── test/
    └── dataset_info.txt
```

## COCO标注格式说明

### JSON结构

```json
{
  "info": {
    "description": "FRED Dataset - RGB Modality",
    "version": "1.0",
    "year": 2024,
    ...
  },
  "licenses": [...],
  "images": [
    {
      "id": 1,
      "file_name": "0_Video_0_16_03_49.204702.jpg",
      "width": 1280,
      "height": 720,
      ...
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  # 左上角坐标和宽高（像素）
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object",
      "supercategory": "none"
    }
  ]
}
```

### 边界框格式

- **YOLO格式** (输入): `[x_center, y_center, width, height]` - 归一化坐标 (0-1)
- **COCO格式** (输出): `[x, y, width, height]` - 像素坐标，(x,y)为左上角

## 使用COCO API验证数据集

安装COCO API：
```bash
pip install pycocotools
```

验证脚本示例：
```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 加载COCO标注
coco = COCO('datasets/fred_coco/rgb/annotations/instances_train.json')

# 获取所有类别
cats = coco.loadCats(coco.getCatIds())
print(f"类别: {[cat['name'] for cat in cats]}")

# 获取所有图片ID
img_ids = coco.getImgIds()
print(f"图片数量: {len(img_ids)}")

# 加载一张图片
img_info = coco.loadImgs(img_ids[0])[0]
print(f"图片信息: {img_info}")

# 获取该图片的所有标注
ann_ids = coco.getAnnIds(imgIds=img_info['id'])
anns = coco.loadAnns(ann_ids)
print(f"标注数量: {len(anns)}")

# 可视化
img = Image.open(f"datasets/fred_coco/rgb/train/{img_info['file_name']}")
plt.imshow(img)
coco.showAnns(anns)
plt.show()
```

## 集成到YOLOv5训练

虽然当前项目使用VOC格式，但可以通过以下方式使用COCO数据：

### 方法1: 使用COCO工具转换为VOC

创建 `coco_to_voc.py` 脚本（见下文）

### 方法2: 修改数据加载器支持COCO

修改 `utils/dataloader.py` 以支持COCO格式的数据加载。

### 方法3: 使用现有的COCO转换工具

```bash
# 使用项目自带的COCO工具
python utils_coco/coco_annotation.py
```

## 数据集质量检查

转换完成后，建议进行以下检查：

1. **检查标注文件**
```bash
# 查看标注统计
python -c "
import json
with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
    data = json.load(f)
    print(f'图片数: {len(data[\"images\"])}')
    print(f'标注数: {len(data[\"annotations\"])}')
    print(f'类别数: {len(data[\"categories\"])}')
"
```

2. **可视化检查**
创建可视化脚本检查标注是否正确

3. **验证边界框**
确保所有边界框都在图像范围内

## 常见问题

### Q1: 为什么有些图片没有标注？
A: FRED数据集中，只有包含目标的帧才有标注。空标注文件（大小为0）的图片会被自动过滤。

### Q2: RGB和Event模态的标注数量为什么不同？
A: 两种模态的采样率和目标可见性不同，导致有效标注数量有差异。

### Q3: 如何修改类别？
A: 修改 `convert_fred_to_coco.py` 中的 `CATEGORIES` 变量。

### Q4: 如何调整数据集划分比例？
A: 使用 `--train_ratio`, `--val_ratio`, `--test_ratio` 参数。

## 下一步

1. **验证转换结果**
```bash
/home/yz/miniforge3/envs/torch/bin/python3 verify_coco_dataset.py --modality rgb
```

2. **创建训练配置**
根据COCO数据集创建对应的训练配置文件

3. **开始训练**
使用转换后的数据集进行模型训练

## 参考资料

- [COCO数据集格式](https://cocodataset.org/#format-data)
- [pycocotools文档](https://github.com/cocodataset/cocoapi)
- [YOLOv5训练指南](./AGENTS.md)
