# YOLOv5-PyTorch 项目指南

## 项目概述

这是一个基于 PyTorch 实现的 YOLOv5 目标检测项目，fork 自 [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch)。该项目提供了完整的训练、预测、评估和数据处理功能，支持多种主干网络和数据增强策略。

### 主要特性

- **框架**: PyTorch 2.4.1
- **CUDA**: 12.4
- **GPU**: NVIDIA GeForce RTX 3090
- **模型**: YOLOv5 (支持 s/m/l/x 版本)
- **主干网络**: 
  - CSPDarknet (默认)
  - ConvNeXt (tiny/small)
  - Swin Transformer (tiny)
- **数据格式**: 
  - VOC 格式 (XML 标注)
  - COCO 格式 (JSON 标注)
- **训练策略**: 支持冻结训练、解冻训练、Mosaic 数据增强、Mixup 数据增强
- **部署**: 支持 ONNX 导出

### 特殊功能

本项目已集成 **FRED 数据集**支持：
- RGB 模态: 19,471 张图片
- Event 模态: 28,714 张图片
- 单类别目标检测
- 小目标优化（平均 50x34 像素）

---

## 项目结构

```
yolov5-pytorch/
├── nets/                    # 网络模型定义
│   ├── yolo.py             # YOLOv5 主网络
│   ├── CSPdarknet.py       # CSPDarknet 主干
│   ├── ConvNext.py         # ConvNeXt 主干
│   ├── Swin_transformer.py # Swin Transformer 主干
│   └── yolo_training.py    # 训练相关函数（损失函数等）
├── utils/                   # 工具函数
│   ├── dataloader.py       # 数据加载器
│   ├── utils_fit.py        # 训练循环
│   ├── utils_map.py        # mAP 计算
│   ├── utils_bbox.py       # 边界框处理
│   └── callbacks.py        # 训练回调
├── utils_coco/             # COCO 数据集工具
├── model_data/             # 模型权重和配置文件
├── logs/                   # 训练日志和权重保存目录
├── img/                    # 测试图片
├── datasets/               # 数据集目录
│   ├── nps/               # NPS 数据集配置
│   └── fred_coco/         # FRED COCO 格式数据集
│       ├── rgb/           # RGB 模态
│       └── event/         # Event 模态
├── train.py               # 标准训练脚本
├── train_fred.py          # FRED 数据集训练脚本
├── predict.py             # 标准预测脚本
├── predict_fred.py        # FRED 数据集预测脚本
├── yolo.py                # YOLO 类定义（用于推理）
├── get_map.py             # mAP 评估脚本
├── eval_fred.py           # FRED 数据集评估脚本
├── voc_annotation.py      # VOC 数据集标注处理
├── convert_fred_to_coco.py # FRED 数据集转换脚本
├── kmeans_for_anchors.py  # K-means 计算先验框
├── summary.py             # 模型结构查看
└── 常见问题汇总.md        # 常见问题解答
```

---

## 环境配置

### 当前环境

```
Python: /home/yz/.conda/envs/torch/bin/python3
PyTorch: 2.4.1
CUDA: 12.4
GPU: NVIDIA GeForce RTX 3090
```

### 依赖项

根据 `requirements.txt`，项目需要以下依赖：

```
torch
torchvision
tensorboard
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```

### 重要版本说明

- **h5py**: 必须为 2.10.0（3.0.0+ 会导致 decode("utf-8") 错误）
- **Pillow**: 建议为 8.2.0（避免 __array__() 错误）
- **PyTorch**: 1.7.1+ 以支持混合精度训练和 ONNX 导出

### 安装步骤

```bash
# 安装依赖
pip install -r requirements.txt

# 强制安装特定版本（重要！）
pip install h5py==2.10.0
pip install Pillow==8.2.0
```

---

## 快速开始

### 1. 标准 VOC 数据集训练

```bash
# 1. 准备数据集（VOC 格式）
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

### 2. FRED 数据集训练（推荐）

```bash
# 1. 确保 FRED 数据集已转换为 COCO 格式
# 数据集位置: datasets/fred_coco/

# 2. 训练 RGB 模态
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality rgb

# 3. 训练 Event 模态
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality event

# 4. 评估模型
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality rgb

# 5. 预测测试
/home/yz/.conda/envs/torch/bin/python3 predict_fred.py --modality rgb
```

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

2. **修改类别文件**: 编辑 `classes_path` 指向的文件（如 `datasets/nps/classes.txt`），每行一个类别名

3. **生成训练文件**: 运行 `voc_annotation.py`
   ```bash
   python voc_annotation.py
   ```
   这将生成：
   - `ImageSets/Main/` 下的 train.txt, val.txt, test.txt
   - 根目录下的 `2007_train.txt`, `2007_val.txt` (包含完整路径和标注信息)

4. **（可选）计算先验框**: 如果数据集目标尺寸分布特殊，可重新计算 anchors
   ```bash
   python kmeans_for_anchors.py
   ```

### COCO 格式数据集（FRED）

#### 数据集结构

```
datasets/fred_coco/
├── rgb/                     # RGB 模态 COCO 数据集
│   ├── annotations/         # JSON 标注文件
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   ├── train/              # 训练集图片 (13,629张)
│   ├── val/                # 验证集图片 (3,894张)
│   ├── test/               # 测试集图片 (1,948张)
│   └── dataset_info.txt    # 数据集信息
│
└── event/                   # Event 模态 COCO 数据集
    ├── annotations/         # JSON 标注文件
    ├── train/              # 训练集图片 (20,099张)
    ├── val/                # 验证集图片 (5,742张)
    ├── test/               # 测试集图片 (2,873张)
    └── dataset_info.txt    # 数据集信息
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

---

## 训练模型

### 标准训练（VOC 数据集）

#### 训练前配置

在 `train.py` 中需要修改的关键参数：

```python
# 数据集配置
classes_path = 'datasets/nps/classes.txt'          # 类别文件
anchors_path = 'datasets/nps/anchors.txt'          # 先验框文件
train_annotation_path = 'datasets/nps/train.txt'   # 训练集
val_annotation_path = 'datasets/nps/val.txt'       # 验证集
test_annotation_path = 'datasets/nps/test.txt'     # 测试集

# 模型配置
model_path = 'model_data/yolov5_s.pth'  # 预训练权重（首次训练）
# model_path = 'logs/best_epoch_weights.pth'  # 断点续练
input_shape = [640, 640]                # 输入尺寸（必须是32的倍数）
backbone = 'cspdarknet'                 # 主干网络
phi = 's'                               # YOLOv5 版本 (s/m/l/x)

# 训练参数
Cuda = True                             # 是否使用 GPU
Init_Epoch = 0                          # 起始轮次
Freeze_Epoch = 50                       # 冻结训练轮次
UnFreeze_Epoch = 300                    # 总训练轮次
Freeze_batch_size = 16                  # 冻结阶段 batch size
Unfreeze_batch_size = 16                # 解冻阶段 batch size
Freeze_Train = True                     # 是否进行冻结训练

# 优化器配置
optimizer_type = "sgd"                  # 优化器类型 (sgd/adam)
Init_lr = 1e-2                          # 初始学习率
Min_lr = Init_lr * 0.01                 # 最小学习率
momentum = 0.937                        # SGD 动量
weight_decay = 5e-4                     # 权重衰减

# 数据增强
mosaic = True                           # Mosaic 数据增强
mosaic_prob = 0.5                       # Mosaic 概率
mixup = True                            # Mixup 数据增强
mixup_prob = 0.5                        # Mixup 概率
special_aug_ratio = 0.7                 # 前70%轮次使用强数据增强
```

#### 训练命令

```bash
# 标准训练
python train.py

# 仅评估模式（不训练）
python train.py --eval_only

# 多 GPU 训练（DDP 模式，仅 Linux）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### FRED 数据集训练

#### 配置文件

FRED 数据集的配置在 `config_fred.py` 中：

```python
# 选择模态
MODALITY = 'rgb'  # 'rgb' 或 'event'

# 模型配置
INPUT_SHAPE = [640, 640]
BACKBONE = 'cspdarknet'
PHI = 's'

# 训练配置
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_BATCH_SIZE = 16
UNFREEZE_BATCH_SIZE = 8  # FRED 数据集目标较小，使用较小的 batch

# 优化器配置
OPTIMIZER_TYPE = 'sgd'
INIT_LR = 1e-2
MIN_LR = 1e-4

# 数据增强
MOSAIC = True
MIXUP = True
```

#### 训练命令

```bash
# RGB 模态（推荐）
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality rgb

# Event 模态
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality event

# 快速训练（不评估 mAP）
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality rgb --no_eval_map

# 使用快捷脚本
bash start_training.sh
```

### 训练策略说明

#### 冻结训练 vs 解冻训练

- **冻结阶段** (0-50 epoch): 
  - 冻结主干网络，仅训练检测头
  - 显存占用小（~6GB）
  - 适合显存不足的情况
  - 每个 epoch 约 5 分钟

- **解冻阶段** (50-300 epoch): 
  - 解冻主干网络，全网络训练
  - 显存占用大（~10GB）
  - 提升模型性能
  - 每个 epoch 约 8 分钟

如果显存不足，可设置 `Freeze_Epoch = UnFreeze_Epoch`，仅进行冻结训练。

#### 从零开始训练 vs 预训练权重

- **推荐**: 使用预训练权重（`model_path` 指向预训练模型）
- **从零训练**: 设置 `model_path = ''` 和 `Freeze_Train = False`
  - 需要更大的数据集（万级以上）
  - 需要更长的训练时间（300+ epochs）
  - 需要更大的 batch size（16+）
  - 建议开启 Mosaic 数据增强

**重要**: 对于 99% 的情况都必须使用预训练权重，不使用会导致训练效果很差！

### 断点续练

```python
# 在 train.py 或 train_fred.py 中设置
model_path = 'logs/best_epoch_weights.pth'  # 或其他已训练的权重
Init_Epoch = 60  # 从第60轮继续
```

### 训练输出

#### 标准训练输出

```
logs/
├── loss_{timestamp}/           # 训练日志
│   ├── events.out.tfevents.*  # TensorBoard 日志
│   ├── epoch_loss.png         # Loss 曲线
│   └── lr.png                 # 学习率曲线
├── best_epoch_weights.pth     # 验证集最佳权重 ⭐
├── last_epoch_weights.pth     # 最后一轮权重
└── ep{epoch}-loss{loss}-val_loss{val_loss}.pth  # 定期保存的权重
```

#### FRED 训练输出

```
logs/fred_rgb/                          # RGB 模态
├── loss_2025_10_23_XX_XX_XX/          # 训练日志
│   ├── events.out.tfevents.*          # TensorBoard
│   ├── epoch_loss.png                 # Loss 曲线
│   └── lr.png                         # 学习率曲线
├── .temp_map_out/                     # mAP 评估结果
│   └── results/
│       └── results.txt                # mAP 详细结果
├── fred_rgb_best.pth                  # 最佳模型 ⭐
├── fred_rgb_final.pth                 # 最终模型
└── ep*-loss*-val_loss*.pth            # 定期保存

logs/fred_event/                        # Event 模态
└── (同上)
```

### 预期训练时间（RTX 3090）

#### FRED RGB 模态（300 epochs）
- 冻结阶段 (50 epochs): ~4 小时
- 解冻阶段 (250 epochs): ~33 小时
- mAP 评估 (30 次): ~1.5 小时
- **总计**: ~38.5 小时

#### FRED Event 模态（300 epochs）
- 冻结阶段 (50 epochs): ~4.5 小时
- 解冻阶段 (250 epochs): ~35 小时
- mAP 评估 (30 次): ~1.5 小时
- **总计**: ~41 小时

---

## 预测推理

### 标准预测（VOC 数据集）

#### 配置预测参数

在 `yolo.py` 中修改 `_defaults` 字典：

```python
_defaults = {
    "model_path": 'logs/best_epoch_weights.pth',  # 模型权重
    "classes_path": 'datasets/nps/classes.txt',   # 类别文件
    "anchors_path": 'datasets/nps/anchors.txt',   # 先验框文件
    "input_shape": [640, 640],                    # 输入尺寸
    "backbone": 'cspdarknet',                     # 主干网络
    "phi": 's',                                   # 模型版本
    "confidence": 0.5,                            # 置信度阈值
    "nms_iou": 0.3,                              # NMS IOU 阈值
    "letterbox_image": True,                      # 是否使用 letterbox
    "cuda": True,                                 # 是否使用 GPU
}
```

#### 预测模式

在 `predict.py` 中设置 `mode` 参数：

```python
# 单张图片预测
mode = "predict"
python predict.py
# 输入图片路径，显示检测结果

# 视频/摄像头检测
mode = "video"
video_path = 0  # 0 表示摄像头，或设置为视频文件路径
video_save_path = "output.mp4"  # 保存路径（空字符串表示不保存）
python predict.py

# FPS 测试
mode = "fps"
fps_image_path = "img/street.jpg"
test_interval = 100
python predict.py

# 批量预测（遍历文件夹）
mode = "dir_predict"
dir_origin_path = "img/"      # 输入文件夹
dir_save_path = "img_out/"    # 输出文件夹
python predict.py

# 热力图可视化
mode = "heatmap"
heatmap_save_path = "model_data/heatmap_vision.png"
python predict.py

# 导出 ONNX
mode = "export_onnx"
onnx_save_path = "model_data/models.onnx"
simplify = True
python predict.py

# 使用 ONNX 预测
mode = "predict_onnx"
python predict.py
```

### FRED 数据集预测

```bash
# RGB 模态预测
/home/yz/.conda/envs/torch/bin/python3 predict_fred.py --modality rgb

# Event 模态预测
/home/yz/.conda/envs/torch/bin/python3 predict_fred.py --modality event
```

---

## 模型评估

### 标准评估（VOC 数据集）

#### 计算 mAP

```bash
python get_map.py
```

#### 评估参数配置

在 `get_map.py` 中：

```python
map_mode = 0           # 0: 完整流程, 1: 仅预测, 2: 仅真实框, 3: 仅计算mAP
classes_path = 'datasets/nps/classes.txt'
MINOVERLAP = 0.5       # mAP@0.5 的 IOU 阈值
confidence = 0.001     # 预测置信度（需要很小以获取所有预测框）
nms_iou = 0.5          # NMS IOU 阈值
score_threhold = 0.5   # 计算 Recall/Precision 的阈值
map_vis = False        # 是否可视化
VOCdevkit_path = 'VOCdevkit'
map_out_path = 'map_out'
```

#### 评估输出

```
map_out/
├── detection-results/  # 预测结果
├── ground-truth/       # 真实标注
└── results/            # mAP 计算结果、PR 曲线等
```

### FRED 数据集评估

```bash
# RGB 模态评估
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality rgb

# Event 模态评估
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality event

# 快速评估（使用快捷脚本）
bash quick_eval.sh
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

可查看：
- 训练/验证损失曲线
- 学习率变化
- mAP 变化（如果启用）

### 方法3: 查看日志文件

```bash
# 查看最新的训练日志
ls -lt logs/

# 查看 FRED mAP 结果
cat logs/fred_rgb/.temp_map_out/results/results.txt
```

---

## 常见问题与解决方案

### 1. 环境问题

#### 显存不足 (OOM / CUDA out of memory)

**解决方案**:
- 减小 `batch_size`
- 减小 `input_shape`
- 选择更小的模型版本（如 yolov5s）
- 注意：`batch_size` 最小为 2（受 BatchNorm 影响）

```python
# 方案1: 减小 batch size
Freeze_batch_size = 8
Unfreeze_batch_size = 4

# 方案2: 减小输入尺寸
input_shape = [416, 416]  # 从 640 降到 416

# 方案3: 禁用 mAP 评估（FRED 训练）
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality rgb --no_eval_map
```

#### No module named 'xxx'

**解决方案**:
- 检查是否激活了正确的 conda 环境
- 使用 `pip install xxx` 安装缺失的库
- 对于项目内部模块（如 utils），检查当前工作目录是否为项目根目录

```bash
# 激活环境
conda activate torch

# 安装缺失的库
pip install matplotlib
pip install opencv-python

# 检查当前目录
pwd  # 应该在 /mnt/data/code/yolov5-pytorch
```

#### DLL load failed / 找不到指定模块

**解决方案**:
- 重启系统
- 重新安装 CUDA、cuDNN、PyTorch
- 检查版本兼容性

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

### 2. 训练问题

#### Shape 不匹配

**问题**: 训练或预测时提示 shape 不匹配

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
4. ✅ 确认训练轮次是否足够（按默认参数训练完）
5. ✅ 检查 `classes_path` 是否正确设置
6. ✅ 检查是否使用了预训练权重

#### 断点续练

```python
# 在 train.py 或 train_fred.py 中设置
model_path = 'logs/best_epoch_weights.pth'  # 已训练的权重路径
Init_Epoch = 60  # 从第 60 轮继续
```

### 3. 数据问题

#### GBK 编码错误

**问题**: 'gbk' codec can't decode byte

**解决方案**:
- 路径和标签中不要使用中文
- 如必须使用中文，修改文件打开方式为 `encoding='utf-8'`

#### 图片分辨率问题

**说明**: 任意分辨率的图片都可以使用，代码会自动 resize

#### 灰度图问题

**说明**: 代码会自动将灰度图转换为 RGB

#### 路径问题 (No such file or directory)

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

#### 如何提升模型效果

**建议**:
1. 增加数据集规模
2. 进行解冻训练
3. 增加训练轮次
4. 调整数据增强策略
5. 尝试不同的主干网络
6. 参考 YOLOv4 论文中的 tricks

---

## 开发约定

### 代码风格

- 遵循项目现有的代码风格
- 使用 4 空格缩进
- 中文注释使用 UTF-8 编码

### 文件路径

- 所有路径使用相对于项目根目录的相对路径
- 数据集路径可配置，默认在项目根目录下
- 使用工具时必须提供绝对路径

### 配置管理

- 标准训练配置在 `train.py` 中修改
- FRED 训练配置在 `config_fred.py` 中修改
- 预测配置在 `yolo.py` 的 `_defaults` 字典中修改
- 评估配置在 `get_map.py` 或 `eval_fred.py` 中修改

### Git 使用

- 不要提交 `logs/` 目录下的权重文件
- 不要提交数据集文件
- 提交前检查 `.gitignore` 配置

---

## 扩展功能

### COCO 数据集支持

使用 `utils_coco/coco_annotation.py` 处理 COCO 格式数据集：

```bash
python utils_coco/coco_annotation.py
```

### FRED 数据集转换

```bash
# 转换 RGB 模态
python convert_fred_to_coco.py --modality rgb

# 转换 Event 模态
python convert_fred_to_coco.py --modality event

# 验证数据集
python verify_coco_dataset.py --modality rgb --split train
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

## 重要文档

### 核心文档

- `README.md` - 项目简介
- `常见问题汇总.md` - 详细的常见问题解答
- `AGENTS.md` - 本文档

### FRED 数据集文档

- `QUICK_START_FRED.md` - FRED 数据集快速开始指南
- `START_TRAINING.md` - FRED 训练详细指南
- `FRED_FINAL_REPORT.md` - FRED 数据集转换报告
- `README_FRED_COCO.md` - FRED COCO 格式详细文档
- `EVAL_GUIDE.md` - 评估指南
- `VISUALIZATION_GUIDE.md` - 可视化指南

### 快速参考

- `QUICK_REFERENCE.md` - 快速参考
- `TRAINING_COMMANDS.md` - 训练命令汇总
- `QUICK_TEST_GUIDE.md` - 快速测试指南

---

## 参考资源

- 原始仓库: https://github.com/bubbliiiing/yolov5-pytorch
- 相关博客: https://blog.csdn.net/weixin_44791964
- 常见问题汇总: 见项目根目录 `常见问题汇总.md`

---

## 注意事项

### 关键要点

1. **预训练权重**: 对于 99% 的情况都必须使用，不使用会导致训练效果很差
2. **数据集大小**: 建议至少 500 张图片，小数据集需要更长的训练时间
3. **显存管理**: 根据显卡显存调整 `batch_size` 和 `input_shape`
4. **训练时长**: SGD 优化器需要更长的训练时间（300+ epochs），Adam 可以更短（100+ epochs）
5. **评估指标**: mAP 是主要评估指标，Loss 仅用于判断收敛
6. **版本兼容**: 注意 PyTorch、CUDA、cuDNN 版本的兼容性
7. **h5py 版本**: 必须使用 2.10.0，否则会出现 decode 错误
8. **Pillow 版本**: 建议使用 8.2.0，避免 __array__() 错误

### FRED 数据集特点

- **单类别**: 只有一个类别 (object)
- **小目标**: 平均边界框 50x34 像素
- **两种模态**: RGB (19,471 张) 和 Event (28,714 张)
- **Event 边界框**: 约 3% 的边界框被裁剪（原始标注超出边界），这是正常的

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
# 1. 训练 RGB 模态
/home/yz/.conda/envs/torch/bin/python3 train_fred.py --modality rgb

# 2. 评估模型
/home/yz/.conda/envs/torch/bin/python3 eval_fred.py --modality rgb

# 3. 预测测试
/home/yz/.conda/envs/torch/bin/python3 predict_fred.py --modality rgb

# 4. 查看训练曲线
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

# 验证 FRED 数据集
python verify_coco_dataset.py --modality rgb --split train

# 可视化 FRED 样本
python visualize_dataset.py --modality rgb --num_samples 10
```

---

**最后更新**: 2025-10-24  
**项目路径**: `/mnt/data/code/yolov5-pytorch`  
**Python 环境**: `/home/yz/.conda/envs/torch/bin/python3`  
**系统配置**: RTX 3090 / CUDA 12.4 / PyTorch 2.4.1
