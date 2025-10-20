# FRED数据集COCO格式转换总结

## 转换完成状态 ✅

已成功将FRED数据集转换为COCO格式，包含RGB和Event两种模态。

## 数据集统计

### RGB模态
- **训练集**: 13,629 张图片，13,629 个标注
- **验证集**: 3,894 张图片，3,894 个标注
- **测试集**: 1,948 张图片，1,948 个标注
- **总计**: 19,471 张图片，19,471 个标注

**边界框统计**:
- 平均宽度: 50.22 像素
- 平均高度: 34.08 像素
- 宽度范围: 13.65 - 662.86 像素
- 高度范围: 11.00 - 638.22 像素

### Event模态
- **训练集**: 20,099 张图片，20,099 个标注
- **验证集**: 5,742 张图片，5,742 个标注
- **测试集**: 2,873 张图片，2,873 个标注
- **总计**: 28,714 张图片，28,714 个标注

**边界框统计**:
- 平均宽度: 50.96 像素
- 平均高度: 34.58 像素
- 宽度范围: 13.65 - 662.86 像素
- 高度范围: 12.65 - 638.22 像素

## 目录结构

```
datasets/fred_coco/
├── rgb/
│   ├── annotations/
│   │   ├── instances_train.json    # 训练集标注 (13,629 images)
│   │   ├── instances_val.json      # 验证集标注 (3,894 images)
│   │   └── instances_test.json     # 测试集标注 (1,948 images)
│   ├── train/                      # 训练集图片 (JPG格式)
│   ├── val/                        # 验证集图片
│   ├── test/                       # 测试集图片
│   └── dataset_info.txt            # 数据集信息
│
└── event/
    ├── annotations/
    │   ├── instances_train.json    # 训练集标注 (20,099 images)
    │   ├── instances_val.json      # 验证集标注 (5,742 images)
    │   └── instances_test.json     # 测试集标注 (2,873 images)
    ├── train/                      # 训练集图片 (PNG格式)
    ├── val/                        # 验证集图片
    ├── test/                       # 测试集图片
    └── dataset_info.txt            # 数据集信息
```

## 数据集特点

1. **单类别检测**: 所有标注都属于 "object" 类别 (category_id=1)
2. **每张图片一个目标**: 平均每张图片包含1个标注
3. **小目标为主**: 平均边界框尺寸约为 50x34 像素
4. **图像尺寸**: 1280x720 像素
5. **数据划分**: 70% 训练 / 20% 验证 / 10% 测试

## 验证结果

✅ **RGB模态**: 所有验证通过
- 图片文件完整
- 标注格式正确
- 边界框有效

✅ **Event模态**: 所有验证通过
- 图片文件完整
- 标注格式正确
- 边界框有效

## 快速使用

### 1. 加载COCO数据集

```python
from pycocotools.coco import COCO

# 加载RGB训练集
coco_rgb_train = COCO('datasets/fred_coco/rgb/annotations/instances_train.json')

# 加载Event训练集
coco_event_train = COCO('datasets/fred_coco/event/annotations/instances_train.json')

# 获取所有图片ID
img_ids = coco_rgb_train.getImgIds()
print(f"训练集图片数: {len(img_ids)}")

# 获取所有类别
cats = coco_rgb_train.loadCats(coco_rgb_train.getCatIds())
print(f"类别: {[cat['name'] for cat in cats]}")
```

### 2. 读取图片和标注

```python
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载第一张图片
img_id = img_ids[0]
img_info = coco_rgb_train.loadImgs(img_id)[0]
img_path = f"datasets/fred_coco/rgb/train/{img_info['file_name']}"
img = Image.open(img_path)

# 获取该图片的标注
ann_ids = coco_rgb_train.getAnnIds(imgIds=img_id)
anns = coco_rgb_train.loadAnns(ann_ids)

# 可视化
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

for ann in anns:
    bbox = ann['bbox']  # [x, y, width, height]
    rect = patches.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

plt.show()
```

### 3. 数据集迭代器

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

class FREDCocoDataset(Dataset):
    def __init__(self, coco_root, split='train', transform=None):
        self.coco_root = coco_root
        self.split = split
        self.transform = transform
        
        # 加载COCO标注
        ann_file = f"{coco_root}/annotations/instances_{split}.json"
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # 创建image_id到annotations的映射
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = f"{self.coco_root}/{self.split}/{img_info['file_name']}"
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 获取标注
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])
        
        # 提取边界框和类别
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # 转换为 [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, labels

# 使用示例
dataset = FREDCocoDataset('datasets/fred_coco/rgb', split='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, boxes, labels in dataloader:
    print(f"Batch images: {images.shape}")
    print(f"Batch boxes: {len(boxes)}")
    break
```

## 与YOLOv5集成

虽然当前项目使用VOC格式，但可以通过以下方式使用COCO数据：

### 方法1: 使用pycocotools直接加载

修改 `utils/dataloader.py` 以支持COCO格式。

### 方法2: 转换为YOLO格式

COCO格式已经包含了所有必要信息，可以轻松转换回YOLO格式用于训练。

### 方法3: 使用现有工具

项目中的 `utils_coco/` 目录包含COCO相关工具。

## 文件清单

生成的文件：
- ✅ `convert_fred_to_coco.py` - COCO格式转换脚本
- ✅ `verify_coco_dataset.py` - 数据集验证脚本
- ✅ `README_FRED_COCO.md` - 详细使用文档
- ✅ `FRED_DATASET_SUMMARY.md` - 本文档

数据集文件：
- ✅ `datasets/fred_coco/rgb/` - RGB模态COCO数据集
- ✅ `datasets/fred_coco/event/` - Event模态COCO数据集

## 下一步建议

1. **数据探索**
   ```bash
   # 查看数据集统计
   /home/yz/miniforge3/envs/torch/bin/python3 -c "
   import json
   with open('datasets/fred_coco/rgb/annotations/instances_train.json') as f:
       data = json.load(f)
       print(f'训练集: {len(data[\"images\"])} 张图片')
       print(f'标注数: {len(data[\"annotations\"])} 个')
   "
   ```

2. **可视化样本**
   ```bash
   # 生成可视化（需要GUI环境）
   /home/yz/miniforge3/envs/torch/bin/python3 verify_coco_dataset.py \
       --modality rgb --split train --show_samples 5
   ```

3. **训练准备**
   - 根据COCO格式修改训练脚本
   - 或将COCO转换为项目所需的VOC格式
   - 配置数据增强策略

4. **性能优化**
   - 考虑数据预处理和缓存
   - 使用多进程数据加载
   - 实现数据增强

## 技术细节

### COCO格式说明

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "0_Video_0_16_03_49.204702.jpg",
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  // 左上角坐标和宽高
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

### 坐标转换

- **YOLO → COCO**: 
  - YOLO: `[x_center_norm, y_center_norm, w_norm, h_norm]`
  - COCO: `[x_pixel, y_pixel, w_pixel, h_pixel]`
  - 转换: `x = (x_center - w/2) * img_width`

## 联系与支持

如有问题，请参考：
- `README_FRED_COCO.md` - 详细文档
- `AGENTS.md` - 项目整体指南
- 运行验证脚本检查数据集完整性

---

**创建时间**: 2025-10-20  
**Python环境**: /home/yz/miniforge3/envs/torch/bin/python3  
**数据源**: /mnt/data/datasets/fred  
**输出目录**: /mnt/data/code/yolov5-pytorch/datasets/fred_coco
