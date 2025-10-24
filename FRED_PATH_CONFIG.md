# FRED 数据集路径配置说明

## 修改概述

为了避免复制图像文件，我们修改了数据集转换和加载方式：

### 1. 数据集转换（`convert_fred_to_coco.py`）

**修改前**：
- 复制图像文件到 `datasets/fred_coco/{modality}/{split}/`
- `file_name` 格式：`seq0_Video_0_16_03_03.363444.jpg`

**修改后**：
- **不复制**图像文件
- `file_name` 格式：`序列0/PADDED_RGB/Video_0_16_03_03.363444.jpg`（相对于 FRED 根目录的路径）

### 2. 数据加载器（`utils/dataloader_coco.py`）

**关键代码**：
```python
# 第 76 行
img_path = os.path.join(self.image_dir, img['file_name'])
```

**工作原理**：
- `self.image_dir` = FRED 数据集根目录（如 `/mnt/data/datasets/fred`）
- `img['file_name']` = 相对路径（如 `序列0/PADDED_RGB/Video_0_16_03_03.363444.jpg`）
- `img_path` = 完整路径（如 `/mnt/data/datasets/fred/序列0/PADDED_RGB/Video_0_16_03_03.363444.jpg`）

### 3. 训练/评估/预测脚本

**修改的文件**：
- `train_fred.py`
- `eval_fred.py`
- `predict_fred.py`

**修改内容**：
```python
# 修改前
train_img_dir = os.path.join(coco_root, 'train')
val_img_dir = os.path.join(coco_root, 'val')
test_img_dir = os.path.join(coco_root, 'test')

# 修改后
fred_root = '/mnt/data/datasets/fred'  # FRED 数据集根目录
train_img_dir = fred_root
val_img_dir = fred_root
test_img_dir = fred_root
```

## 配置 FRED 数据集路径

### 方法 1: 修改脚本中的硬编码路径（当前方式）

在以下文件中修改 `fred_root` 变量：

### `train_fred.py` (第 71 行)
```python
fred_root = '/home/yz/datasets/fred'  # ← 这里设置 FRED 根目录
```

### `eval_fred.py` (第 45 行)
```python
fred_root = '/home/yz/datasets/fred'  # ← 这里设置 FRED 根目录
```

### `predict_fred.py` (第 48 行)
```python
fred_root = '/home/yz/datasets/fred'  # ← 这里设置 FRED 根目录
```

### 方法 2: 使用环境变量（推荐）

**优点**：更灵活，不需要修改代码

**步骤**：

1. 设置环境变量：
   ```bash
   export FRED_ROOT=/mnt/data/datasets/fred
   ```

2. 修改脚本读取环境变量：
   ```python
   fred_root = os.environ.get('FRED_ROOT', '/mnt/data/datasets/fred')
   ```

3. 在 `~/.bashrc` 或 `~/.zshrc` 中添加（永久生效）：
   ```bash
   export FRED_ROOT=/mnt/data/datasets/fred
   ```

### 方法 3: 使用命令行参数

修改脚本添加 `--fred_root` 参数：

```python
parser.add_argument('--fred_root', type=str, 
                    default='/mnt/data/datasets/fred',
                    help='FRED 数据集根目录')
```

使用时：
```bash
python train_fred.py --modality rgb --fred_root /path/to/fred
```

## 目录结构

### FRED 原始数据集
```
/mnt/data/datasets/fred/
├── 序列0/
│   ├── PADDED_RGB/
│   │   └── Video_0_16_03_03.363444.jpg
│   ├── Event/
│   │   └── Frames/
│   │       └── Video_0_frame_100032333.png
│   └── coordinates.txt
├── 序列1/
│   └── ...
└── ...
```

### COCO 标注文件
```
datasets/fred_coco/
├── rgb/
│   └── annotations/
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
└── event/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
```

### COCO JSON 示例
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "序列0/PADDED_RGB/Video_0_16_03_03.363444.jpg",
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 50, 34],
      "area": 1700
    }
  ]
}
```

## 优点

1. **节省存储空间**：不复制图像文件，节省约 2 倍空间
2. **避免文件名冲突**：使用完整相对路径，不同序列的同名文件不会冲突
3. **符合 COCO 规范**：`file_name` 仍然是相对路径，符合标准
4. **易于维护**：修改原始数据集后，COCO 标注仍然有效

## 注意事项

1. **路径配置**：确保所有脚本中的 `fred_root` 指向正确的 FRED 数据集根目录
2. **依赖原始数据**：删除或移动 FRED 原始数据集会导致训练失败
3. **跨机器使用**：在不同机器上使用时，需要修改 `fred_root` 路径

## 使用流程

### 1. 转换数据集
```bash
# 转换 RGB 模态（不复制图像）
python convert_fred_to_coco.py --modality rgb

# 转换 Event 模态（不复制图像）
python convert_fred_to_coco.py --modality event
```

### 2. 训练模型
```bash
# 确保 train_fred.py 中的 fred_root 路径正确
python train_fred.py --modality rgb
```

### 3. 评估模型
```bash
# 确保 eval_fred.py 中的 fred_root 路径正确
python eval_fred.py --modality rgb
```

### 4. 预测测试
```bash
# 确保 predict_fred.py 中的 fred_root 路径正确
python predict_fred.py --modality rgb
```

---

**最后更新**: 2025-10-24  
**修改原因**: 避免复制图像文件，节省存储空间
