# YOLOv5-PyTorch 项目

> 基于 PyTorch 实现的 YOLOv5 目标检测项目，支持 FRED 数据集训练

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [数据集验证](#数据集验证)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型预测](#模型预测)
- [可视化工具](#可视化工具)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [参考资源](#参考资源)

---

## 项目概述

这是一个基于 PyTorch 实现的 YOLOv5 目标检测项目，fork 自 [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch)。

### 主要特性

- **框架**: PyTorch 2.4.1 + CUDA 12.4
- **GPU**: NVIDIA GeForce RTX 3090
- **模型**: YOLOv5 (s/m/l/x 版本)
- **主干网络**: CSPDarknet / ConvNeXt / Swin Transformer
- **数据格式**: VOC (XML) / COCO (JSON)
- **训练策略**: 冻结训练、解冻训练、Mosaic、Mixup、FP16混合精度
- **部署**: 支持 ONNX 导出

### FRED 数据集支持

本项目已集成 **FRED 数据集**支持：
- **RGB 模态**: 19,471 张图片
- **Event 模态**: 28,714 张图片
- **单类别目标检测**
- **小目标优化** (平均 50x34 像素)

---

## 快速开始

### 1. 标准 VOC 数据集

```bash
# 1. 准备数据集
# 将图片放入 VOCdevkit/VOC2007/JPEGImages/
# 将标注放入 VOCdevkit/VOC2007/Annotations/

# 2. 修改类别文件
# 编辑 datasets/nps/classes.txt

# 3. 生成训练文件
python voc_annotation.py

# 4. 开始训练
python train.py

# 5. 预测测试
python predict.py

# 6. 评估模型
python get_map.py
```

### 2. FRED 数据集（推荐）

#### 一键设置（最简单）

```bash
# 自动完成所有设置
./setup_fred_dataset.sh
```

#### 手动设置

```bash
# 1. 转换数据集为 COCO 格式
python convert_fred_to_coco_v2.py --modality both

# 2. 验证数据集
python test_conversion_v2.py

# 3. 训练 RGB 模态
python train_fred.py --modality rgb

# 4. 评估模型
python eval_fred.py --modality rgb

# 5. 预测测试
python predict_fred.py --modality rgb
```

---

## 环境配置

### 系统要求

- **Python**: 3.6+
- **PyTorch**: 1.7.1+ (推荐 2.4.1)
- **CUDA**: 根据显卡选择
  - 20 系列及以下: CUDA 10.x
  - 30 系列: CUDA 11.0+
  - 40 系列: CUDA 12.x

### 安装依赖

```bash
# 安装依赖
pip install -r requirements.txt

# 强制安装特定版本（重要！）
pip install h5py==2.10.0
pip install Pillow==8.2.0
```

### 依赖项说明

```
torch                    # PyTorch 深度学习框架
torchvision             # PyTorch 视觉库
tensorboard             # 训练可视化
scipy==1.2.1            # 科学计算
numpy==1.17.0           # 数值计算
matplotlib==3.1.2       # 绘图
opencv_python==4.1.2.30 # 图像处理
tqdm==4.60.0            # 进度条
Pillow==8.2.0           # 图像处理（必须 8.2.0）
h5py==2.10.0            # HDF5 文件（必须 2.10.0）
```

**重要版本说明**:
- `h5py==2.10.0`: 必须使用此版本，3.0.0+ 会导致 decode("utf-8") 错误
- `Pillow==8.2.0`: 建议使用此版本，避免 __array__() 错误

---

## 数据准备

### VOC 格式数据集

#### 数据集结构

```
VOCdevkit/
└── VOC2007/
    ├── Annotations/        # XML 标注文件
    │   ├── 000001.xml
    │   └── ...
    ├── JPEGImages/         # 图片文件
    │   ├── 000001.jpg
    │   └── ...
    └── ImageSets/
        └── Main/
            ├── train.txt
            ├── val.txt
            └── test.txt
```

#### 准备步骤

1. **准备数据**: 将图片放入 `JPEGImages/`，XML 标注放入 `Annotations/`

2. **修改类别文件**: 编辑 `datasets/nps/classes.txt`，每行一个类别名

3. **生成训练文件**:
   ```bash
   python voc_annotation.py
   ```

4. **（可选）计算先验框**:
   ```bash
   python kmeans_for_anchors.py
   ```

### COCO 格式数据集（FRED）

#### 数据集结构

```
datasets/fred_coco/
├── rgb/                     # RGB 模态
│   ├── annotations/         # JSON 标注文件
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   ├── train/              # 训练集 (13,629张)
│   ├── val/                # 验证集 (3,894张)
│   └── test/               # 测试集 (1,948张)
│
└── event/                   # Event 模态
    ├── annotations/         # JSON 标注文件
    ├── train/              # 训练集 (20,099张)
    ├── val/                # 验证集 (5,742张)
    └── test/               # 测试集 (2,873张)
```

#### 数据集统计

| 特性 | RGB 模态 | Event 模态 |
|------|---------|-----------|
| 图片格式 | JPG | PNG |
| 图片尺寸 | 1280 x 720 | 1280 x 720 |
| 训练集 | 13,629 张 | 20,099 张 |
| 验证集 | 3,894 张 | 5,742 张 |
| 测试集 | 1,948 张 | 2,873 张 |
| 平均边界框 | 50.22 x 34.08 px | 50.96 x 34.58 px |
| 类别数量 | 1 (object) | 1 (object) |

#### FRED 数据集转换

```bash
# 使用新版转换脚本（推荐）
python convert_fred_to_coco_v2.py --modality both

# 仅转换 RGB 模态
python convert_fred_to_coco_v2.py --modality rgb

# 仅转换 Event 模态
python convert_fred_to_coco_v2.py --modality event

# 验证转换结果
python test_conversion_v2.py
```

---

## 数据集验证

在训练之前，强烈建议先验证数据集，确保标注正确。

### 快速验证（推荐）

```bash
# 使用交互式脚本
bash scripts/validate_dataset.sh

# 然后选择验证模式:
# 1) RGB训练集 (20个样本)
# 2) RGB验证集 (20个样本)
# 3) RGB测试集 (20个样本)
# 4) Event训练集 (20个样本)
# 5) Event验证集 (20个样本)
# 6) Event测试集 (20个样本)
# 7) RGB所有划分 (train/val/test)
# 8) Event所有划分 (train/val/test)
# 9) 全部验证 (RGB+Event, train/val/test)
```

### 命令行验证

```bash
# 验证 RGB 训练集（20个样本）
python scripts/visualize_dataset_validation.py --modality rgb --split train --num_samples 20

# 验证 Event 验证集（10个样本）
python scripts/visualize_dataset_validation.py --modality event --split val --num_samples 10

# 验证所有数据集
python scripts/visualize_dataset_validation.py --modality both --split all --num_samples 20
```

### 验证输出

验证完成后，会在 `dataset_validation/` 目录下生成：

```
dataset_validation/
├── rgb_train/
│   ├── 0001_*.jpg                 # 可视化图片（带边界框）
│   ├── validation_report.json     # JSON 格式验证报告
│   └── validation_report.html     # HTML 格式验证报告 ⭐
├── rgb_val/
├── rgb_test/
├── event_train/
├── event_val/
└── event_test/
```

### 查看验证结果

```bash
# 在浏览器中打开 HTML 报告（推荐）
firefox dataset_validation/rgb_train/validation_report.html

# 或直接查看可视化图片
ls dataset_validation/rgb_train/*.jpg
```

---

## 模型训练

### 标准训练（VOC 数据集）

#### 配置参数

在 `train.py` 中修改关键参数：

```python
# 数据集配置
classes_path = 'datasets/nps/classes.txt'
anchors_path = 'datasets/nps/anchors.txt'
train_annotation_path = 'datasets/nps/train.txt'

# 模型配置
model_path = 'model_data/yolov5_s.pth'  # 预训练权重
input_shape = [640, 640]                # 输入尺寸
backbone = 'cspdarknet'                 # 主干网络
phi = 's'                               # YOLOv5 版本

# 训练参数
Cuda = True
Init_Epoch = 0
Freeze_Epoch = 50                       # 冻结训练轮次
UnFreeze_Epoch = 300                    # 总训练轮次
Freeze_batch_size = 16
Unfreeze_batch_size = 16

# 优化器配置
optimizer_type = "sgd"
Init_lr = 1e-2
Min_lr = Init_lr * 0.01

# 数据增强
mosaic = True
mixup = True
```

#### 训练命令

```bash
# 标准训练
python train.py

# 多 GPU 训练（仅 Linux）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### FRED 数据集训练

#### 配置文件

编辑 `config_fred.py`：

```python
# 选择模态
MODALITY = 'rgb'  # 'rgb' 或 'event'

# 模型配置
INPUT_SHAPE = [640, 640]
BACKBONE = 'cspdarknet'
PHI = 's'

# 训练配置（优化后）
FREEZE_EPOCH = 30           # 50 -> 30
UNFREEZE_EPOCH = 150        # 300 -> 150
FREEZE_BATCH_SIZE = 32      # 16 -> 32
UNFREEZE_BATCH_SIZE = 16    # 8 -> 16

# 优化器配置
OPTIMIZER_TYPE = 'sgd'
INIT_LR = 1e-2
MIN_LR = 1e-4

# 数据增强
MOSAIC = True
MIXUP = False               # True -> False（大数据集不需要）

# 性能优化
FP16 = True                 # 启用混合精度训练
NUM_WORKERS = 8             # 数据加载线程数
```

#### 训练命令

```bash
# RGB 模态（推荐）
python train_fred.py --modality rgb

# Event 模态
python train_fred.py --modality event

# 快速训练（不评估 mAP）
python train_fred.py --modality rgb --no_eval_map

# 快速验证（仅2个epoch）
python train_fred.py --modality rgb --quick_test
```

### 训练策略

#### 冻结训练 vs 解冻训练

- **冻结阶段** (0-30 epoch):
  - 冻结主干网络，仅训练检测头
  - 显存占用小（~8GB）
  - 每个 epoch 约 2.5 分钟

- **解冻阶段** (30-150 epoch):
  - 解冻主干网络，全网络训练
  - 显存占用大（~14GB）
  - 每个 epoch 约 4.5 分钟

#### 预训练权重

**重要**: 对于 99% 的情况都必须使用预训练权重，不使用会导致训练效果很差！

- **推荐**: 使用预训练权重（`model_path` 指向预训练模型）
- **从零训练**: 设置 `model_path = ''` 和 `Freeze_Train = False`
  - 需要更大的数据集（万级以上）
  - 需要更长的训练时间（300+ epochs）
  - 需要更大的 batch size（16+）

### 断点续练

```python
# 在 train.py 或 train_fred.py 中设置
model_path = 'logs/best_epoch_weights.pth'
Init_Epoch = 60  # 从第60轮继续
```

### 训练输出

```
logs/
├── loss_{timestamp}/           # 训练日志
│   ├── events.out.tfevents.*  # TensorBoard 日志
│   ├── epoch_loss.png         # Loss 曲线
│   └── lr.png                 # 学习率曲线
├── best_epoch_weights.pth     # 验证集最佳权重 ⭐
├── last_epoch_weights.pth     # 最后一轮权重
└── ep{epoch}-loss{loss}-val_loss{val_loss}.pth
```

### 预期训练时间（RTX 3090）

#### FRED RGB 模态（优化后，150 epochs）
- 冻结阶段 (30 epochs): ~1.25 小时
- 解冻阶段 (120 epochs): ~9 小时
- mAP 评估 (10 次): ~0.5 小时
- **总计**: ~10.75 小时（原 38.5 小时，加速 3.6 倍）

---

## 模型评估

### 标准评估（VOC 数据集）

```bash
python get_map.py
```

#### 评估参数

在 `get_map.py` 中配置：

```python
map_mode = 0           # 0: 完整流程
classes_path = 'datasets/nps/classes.txt'
MINOVERLAP = 0.5       # mAP@0.5 的 IOU 阈值
confidence = 0.001     # 预测置信度
nms_iou = 0.5          # NMS IOU 阈值
```

### FRED 数据集评估

```bash
# RGB 模态评估
python eval_fred.py --modality rgb

# Event 模态评估
python eval_fred.py --modality event

# 快速评估
bash quick_eval.sh
```

### 评估输出

```
map_out/
├── detection-results/  # 预测结果
├── ground-truth/       # 真实标注
└── results/            # mAP 计算结果、PR 曲线
```

---

## 模型预测

### 标准预测（VOC 数据集）

#### 配置参数

在 `yolo.py` 中修改 `_defaults` 字典：

```python
_defaults = {
    "model_path": 'logs/best_epoch_weights.pth',
    "classes_path": 'datasets/nps/classes.txt',
    "anchors_path": 'datasets/nps/anchors.txt',
    "input_shape": [640, 640],
    "backbone": 'cspdarknet',
    "phi": 's',
    "confidence": 0.5,
    "nms_iou": 0.3,
    "cuda": True,
}
```

#### 预测模式

在 `predict.py` 中设置 `mode` 参数：

```python
# 单张图片预测
mode = "predict"

# 视频/摄像头检测
mode = "video"
video_path = 0  # 0 表示摄像头

# FPS 测试
mode = "fps"

# 批量预测
mode = "dir_predict"
dir_origin_path = "img/"
dir_save_path = "img_out/"

# 导出 ONNX
mode = "export_onnx"
```

### FRED 数据集预测

```bash
# RGB 模态预测
python predict_fred.py --modality rgb

# Event 模态预测
python predict_fred.py --modality event
```

---

## 可视化工具

### FRED 序列可视化

#### 快速生成视频（推荐）

```bash
# 快速预览（前100帧）
./quick_visualize.sh 0 rgb 100

# 完整序列
./quick_visualize.sh 0 rgb

# Event 模态
./quick_visualize.sh 0 event 100
```

#### 使用 Python 脚本

```bash
# 快速预览（50帧，约2秒）
python scripts/visualize_fred_sequences.py \
    --modality rgb \
    --sequence 0 \
    --export-video \
    --no-window \
    --max-frames 50

# 完整序列（1316帧，约15秒）
python scripts/visualize_fred_sequences.py \
    --modality rgb \
    --sequence 21 \
    --export-video \
    --no-window

# RGB 和 Event 对比视频
python scripts/visualize_fred_sequences.py \
    --comparison \
    --sequence 0
```

#### 性能表现（RTX 3090）

| 序列 | 帧数 | 处理时间 | 速度 | 视频大小 |
|------|------|---------|------|---------|
| 序列 0（预览） | 50 | 1.3秒 | 38 FPS | 1.2 MB |
| 序列 21（完整） | 1316 | 14.9秒 | 93 FPS | 9.9 MB |
| 序列 0（完整） | 1909 | ~20秒 | 95 FPS | ~15 MB |

### 可视化特定图片

```bash
# 使用交互式脚本（推荐）
bash scripts/visualize_images.sh

# 或直接使用命令行
# 通过图片ID可视化
python scripts/visualize_specific_images.py --modality rgb --split train --image_ids 1 2 3

# 通过文件名可视化
python scripts/visualize_specific_images.py --modality rgb --split train --filenames Video_0_16_03_03

# 通过正则表达式可视化
python scripts/visualize_specific_images.py --modality rgb --split train --pattern "Video_0_16_03_.*"
```

---

## 监控训练

### 方法1: 实时查看训练输出

训练过程会实时显示：
- 当前 epoch 和进度
- 训练损失和验证损失
- 学习率
- 每个 epoch 的耗时
- mAP 评估结果（如果启用）

### 方法2: TensorBoard

```bash
# 标准训练
tensorboard --logdir logs/

# FRED 训练
tensorboard --logdir logs/fred_rgb/

# 浏览器访问
http://localhost:6006
```

### 方法3: 查看日志文件

```bash
# 查看最新的训练日志
ls -lt logs/

# 查看 FRED mAP 结果
cat logs/fred_rgb/.temp_map_out/results/results.txt
```

---

## 常见问题

### 1. 环境问题

#### 显存不足 (OOM)

**解决方案**:
```python
# 方案1: 减小 batch size
Freeze_batch_size = 8
Unfreeze_batch_size = 4

# 方案2: 减小输入尺寸
input_shape = [416, 416]

# 方案3: 启用 FP16
FP16 = True

# 方案4: 禁用 mAP 评估（FRED 训练）
python train_fred.py --modality rgb --no_eval_map
```

**注意**: `batch_size` 最小为 2（受 BatchNorm 影响）

#### h5py 版本问题

**问题**: 提示 decode("utf-8") 错误

**解决方案**:
```bash
pip install h5py==2.10.0
```

#### Pillow 版本问题

**问题**: 提示 TypeError: __array__() takes 1 positional argument but 2 were given

**解决方案**:
```bash
pip install Pillow==8.2.0
```

#### No module named 'xxx'

**解决方案**:
- 检查是否激活了正确的 conda 环境
- 使用 `pip install xxx` 安装缺失的库
- 对于项目内部模块（如 utils），检查当前工作目录是否为项目根目录

### 2. 训练问题

#### Shape 不匹配

**原因**:
1. 训练时的 `classes_path` 和 `num_classes` 不正确
2. 预测时的 `model_path`、`classes_path` 与训练时不一致

**解决方案**:
- 检查 `train.py` 中的 `classes_path` 和 `num_classes`
- 检查 `yolo.py` 中的 `model_path`、`classes_path`
- 确保训练和预测使用相同的配置

#### Loss 很大/很小

**说明**:
- Loss 的绝对值不重要，重要的是是否收敛（持续下降）
- 不同网络的 Loss 计算方式不同，无法横向比较
- 关注 Loss 是否在变小，预测是否有效果

#### 训练后没有预测结果 / mAP 为 0

**检查清单**:
1. ✅ 检查 `2007_train.txt` 是否有目标信息
2. ✅ 检查数据集大小（建议 > 500 张）
3. ✅ 确认是否进行了解冻训练
4. ✅ 确认训练轮次是否足够
5. ✅ 检查 `classes_path` 是否正确设置
6. ✅ 检查是否使用了预训练权重

### 3. 数据问题

#### GBK 编码错误

**解决方案**:
- 路径和标签中不要使用中文
- 如必须使用中文，修改文件打开方式为 `encoding='utf-8'`

#### 路径问题

**检查清单**:
- ✅ 检查文件夹路径是否正确
- ✅ 检查 `2007_train.txt` 中的路径是否正确
- ✅ 文件夹名称中不要有空格
- ✅ 注意相对路径和绝对路径的区别

**重要**: 所有的路径问题基本上都是根目录问题，好好查一下相对目录的概念！

### 4. 性能问题

#### 检测速度慢

**检查清单**:
- ✅ 检查是否正确安装了 GPU 版本的 PyTorch
- ✅ 使用 `nvidia-smi` 查看 GPU 是否被使用
- ✅ 考虑使用更小的模型或更小的输入尺寸

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看 PyTorch 是否使用 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 项目结构

```
yolov5-pytorch/
├── nets/                    # 网络模型定义
│   ├── yolo.py             # YOLOv5 主网络
│   ├── CSPdarknet.py       # CSPDarknet 主干
│   ├── ConvNext.py         # ConvNeXt 主干
│   ├── Swin_transformer.py # Swin Transformer 主干
│   └── yolo_training.py    # 训练相关函数
├── utils/                   # 工具函数
│   ├── dataloader.py       # 数据加载器
│   ├── utils_fit.py        # 训练循环
│   ├── utils_map.py        # mAP 计算
│   └── callbacks_coco.py   # COCO 数据集回调
├── utils_coco/             # COCO 数据集工具
├── model_data/             # 模型权重和配置文件
├── logs/                   # 训练日志和权重保存目录
├── datasets/               # 数据集目录
│   ├── nps/               # NPS 数据集配置
│   └── fred_coco/         # FRED COCO 格式数据集
│       ├── rgb/           # RGB 模态
│       └── event/         # Event 模态
├── scripts/               # 工具脚本
│   ├── visualize_dataset_validation.py  # 数据集验证可视化
│   ├── visualize_specific_images.py     # 特定图片可视化
│   ├── visualize_fred_sequences.py      # FRED 序列可视化
│   ├── validate_dataset.sh              # 数据集验证快捷脚本
│   ├── visualize_images.sh              # 图片可视化快捷脚本
│   ├── start_training.sh                # 训练快捷脚本
│   ├── quick_eval.sh                    # 评估快捷脚本
│   └── README.md                        # 脚本说明文档
├── train.py               # 标准训练脚本
├── train_fred.py          # FRED 数据集训练脚本
├── predict.py             # 标准预测脚本
├── predict_fred.py        # FRED 数据集预测脚本
├── eval_fred.py           # FRED 数据集评估脚本
├── get_map.py             # mAP 评估脚本
├── config_fred.py         # FRED 训练配置
├── convert_fred_to_coco_v2.py  # FRED 数据集转换脚本 V2
├── quick_visualize.sh     # 快速可视化脚本
└── README.md              # 本文档
```

---

## 扩展功能

### COCO 数据集支持

```bash
python utils_coco/coco_annotation.py
```

### 模型结构查看

```bash
python summary.py
```

### 自定义主干网络

1. 在 `nets/` 目录下添加新的主干网络文件
2. 在 `nets/yolo.py` 中导入并集成
3. 修改 `train.py` 和 `yolo.py` 中的 `backbone` 参数

---

## 参考资源

- **原始仓库**: https://github.com/bubbliiiing/yolov5-pytorch
- **相关博客**: https://blog.csdn.net/weixin_44791964
- **常见问题汇总**: https://blog.csdn.net/weixin_44791964/article/details/107517428

---

## 注意事项

### 关键要点

1. **预训练权重**: 对于 99% 的情况都必须使用，不使用会导致训练效果很差
2. **数据集大小**: 建议至少 500 张图片，小数据集需要更长的训练时间
3. **显存管理**: 根据显卡显存调整 `batch_size` 和 `input_shape`
4. **训练时长**: SGD 优化器需要更长的训练时间（150+ epochs）
5. **评估指标**: mAP 是主要评估指标，Loss 仅用于判断收敛
6. **版本兼容**: 注意 PyTorch、CUDA、cuDNN 版本的兼容性
7. **h5py 版本**: 必须使用 2.10.0
8. **Pillow 版本**: 建议使用 8.2.0
9. **数据验证**: 训练前务必验证数据集标注正确性

### FRED 数据集特点

- **单类别**: 只有一个类别 (object)
- **小目标**: 平均边界框 50x34 像素
- **两种模态**: RGB (19,471 张) 和 Event (28,714 张)
- **Event 边界框**: 约 3% 的边界框被裁剪（原始标注超出边界），这是正常的
- **性能优化**: 使用 FP16 混合精度训练可加速 3.6 倍

---

## 快速命令参考

### 标准训练流程

```bash
# 1. 生成训练文件
python voc_annotation.py

# 2. 训练模型
python train.py

# 3. 预测测试
python predict.py

# 4. 评估模型
python get_map.py
```

### FRED 训练流程

```bash
# 1. 一键设置（推荐）
./setup_fred_dataset.sh

# 2. 或手动转换数据集
python convert_fred_to_coco_v2.py --modality both

# 3. 验证数据集
bash scripts/validate_dataset.sh

# 4. 快速验证训练（2个epoch）
python train_fred.py --modality rgb --quick_test

# 5. 完整训练 RGB 模态
python train_fred.py --modality rgb

# 6. 评估模型
python eval_fred.py --modality rgb

# 7. 预测测试
python predict_fred.py --modality rgb

# 8. 查看训练曲线
tensorboard --logdir logs/fred_rgb/
```

### 常用工具

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看模型结构
python summary.py

# 计算先验框
python kmeans_for_anchors.py

# 验证数据集
bash scripts/validate_dataset.sh

# 可视化 FRED 序列
./quick_visualize.sh 0 rgb 100

# 可视化特定图片
bash scripts/visualize_images.sh
```

---

## License

本项目基于 MIT 协议开源。

---

**最后更新**: 2025-11-01  
**项目路径**: `/mnt/data/code/yolov5-pytorch`  
**Python 环境**: `/home/yz/miniforge3/envs/torch/bin/python3`  
**系统配置**: RTX 3090 / CUDA 12.4 / PyTorch 2.4.1
