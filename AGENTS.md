# YOLOv5-PyTorch 项目指南

## 项目概述

这是一个基于 PyTorch 实现的 YOLOv5 目标检测项目，fork 自 [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch)。该项目提供了完整的训练、预测、评估和数据处理功能，支持多种主干网络和数据增强策略。

### 主要特性

- **框架**: PyTorch
- **模型**: YOLOv5 (支持 s/m/l/x 版本)
- **主干网络**: 
  - CSPDarknet (默认)
  - ConvNeXt (tiny/small)
  - Swin Transformer (tiny)
- **数据格式**: VOC 格式 (XML 标注)
- **训练策略**: 支持冻结训练、解冻训练、Mosaic 数据增强、Mixup 数据增强
- **部署**: 支持 ONNX 导出

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
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── yolo.py                # YOLO 类定义（用于推理）
├── get_map.py             # mAP 评估脚本
├── voc_annotation.py      # VOC 数据集标注处理
├── kmeans_for_anchors.py  # K-means 计算先验框
└── summary.py             # 模型结构查看
```

## 环境配置

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

### 推荐环境

- **PyTorch**: 1.2+ (推荐 1.7.1+ 以支持混合精度训练)
- **CUDA**: 根据显卡选择
  - 20 系列及以下: CUDA 10.x
  - 30 系列: CUDA 11.0+
- **Python**: 3.6+

### 安装步骤

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：h5py 版本必须为 2.10.0
pip install h5py==2.10.0

# 注意：Pillow 版本建议为 8.2.0
pip install Pillow==8.2.0
```

## 数据准备

### 数据集格式

项目使用 VOC 格式数据集：

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

### 数据集准备步骤

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

## 训练模型

### 训练前配置

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

### 训练命令

```bash
# 标准训练
python train.py

# 仅评估模式（不训练）
python train.py --eval_only

# 多 GPU 训练（DDP 模式，仅 Linux）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 训练策略说明

#### 冻结训练 vs 解冻训练

- **冻结阶段** (0-50 epoch): 冻结主干网络，仅训练检测头，显存占用小，适合显存不足的情况
- **解冻阶段** (50-300 epoch): 解冻主干网络，全网络训练，显存占用大，提升模型性能

如果显存不足，可设置 `Freeze_Epoch = UnFreeze_Epoch`，仅进行冻结训练。

#### 从零开始训练 vs 预训练权重

- **推荐**: 使用预训练权重（`model_path` 指向预训练模型）
- **从零训练**: 设置 `model_path = ''` 和 `Freeze_Train = False`
  - 需要更大的数据集（万级以上）
  - 需要更长的训练时间（300+ epochs）
  - 需要更大的 batch size（16+）
  - 建议开启 Mosaic 数据增强

### 断点续练

```python
# 在 train.py 中设置
model_path = 'logs/best_epoch_weights.pth'  # 或其他已训练的权重
Init_Epoch = 60  # 从第60轮继续
```

### 训练输出

- **权重文件**: 保存在 `logs/` 目录
  - `best_epoch_weights.pth`: 验证集最佳权重
  - `last_epoch_weights.pth`: 最后一轮权重
  - `ep{epoch}-loss{loss}-val_loss{val_loss}.pth`: 定期保存的权重
- **日志文件**: 保存在 `logs/loss_{timestamp}/` 目录
  - TensorBoard 日志
  - Loss 曲线
  - mAP 曲线

## 预测推理

### 配置预测参数

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

### 预测模式

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

## 模型评估

### 计算 mAP

```bash
python get_map.py
```

### 评估参数配置

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

### 评估输出

- `map_out/detection-results/`: 预测结果
- `map_out/ground-truth/`: 真实标注
- `map_out/results/`: mAP 计算结果、PR 曲线等

## 常见问题与解决方案

### 1. 环境问题

**Q: 显存不足 (OOM / CUDA out of memory)**
- 减小 `batch_size`
- 减小 `input_shape`
- 选择更小的模型版本（如 yolov5s）
- 注意：`batch_size` 最小为 2（受 BatchNorm 影响）

**Q: No module named 'xxx'**
- 检查是否激活了正确的 conda 环境
- 使用 `pip install xxx` 安装缺失的库
- 对于项目内部模块（如 utils），检查当前工作目录是否为项目根目录

**Q: DLL load failed / 找不到指定模块**
- 重启系统
- 重新安装 CUDA、cuDNN、PyTorch
- 检查版本兼容性

### 2. 训练问题

**Q: Shape 不匹配**
- 训练时：检查 `train.py` 中的 `classes_path` 和 `num_classes` 是否正确
- 预测时：检查 `yolo.py` 中的 `model_path`、`classes_path` 是否与训练时一致

**Q: Loss 很大/很小**
- Loss 的绝对值不重要，重要的是是否收敛（持续下降）
- 不同网络的 Loss 计算方式不同，无法横向比较

**Q: 训练后没有预测结果 / mAP 为 0**
1. 检查 `2007_train.txt` 是否有目标信息
2. 检查数据集大小（建议 > 500 张）
3. 确认是否进行了解冻训练
4. 确认训练轮次是否足够（按默认参数训练完）
5. 检查 `classes_path` 是否正确设置
6. 检查是否使用了预训练权重

**Q: 如何进行断点续练？**
- 设置 `model_path` 为已训练的权重路径（如 `logs/best_epoch_weights.pth`）
- 调整 `Init_Epoch` 为继续的起始轮次

### 3. 数据问题

**Q: GBK 编码错误**
- 路径和标签中不要使用中文
- 如必须使用中文，修改文件打开方式为 `encoding='utf-8'`

**Q: 图片分辨率问题**
- 任意分辨率的图片都可以使用，代码会自动 resize

**Q: 灰度图问题**
- 代码会自动将灰度图转换为 RGB

**Q: 路径问题 (No such file or directory)**
- 检查文件夹路径是否正确
- 检查 `2007_train.txt` 中的路径是否正确
- 文件夹名称中不要有空格
- 注意相对路径和绝对路径的区别

### 4. 性能问题

**Q: 检测速度慢**
- 检查是否正确安装了 GPU 版本的 PyTorch
- 使用 `nvidia-smi` 查看 GPU 是否被使用
- 考虑使用更小的模型或更小的输入尺寸

**Q: 如何提升模型效果？**
1. 增加数据集规模
2. 进行解冻训练
3. 增加训练轮次
4. 调整数据增强策略
5. 尝试不同的主干网络
6. 参考 YOLOv4 论文中的 tricks

## 开发约定

### 代码风格

- 遵循项目现有的代码风格
- 使用 4 空格缩进
- 中文注释使用 UTF-8 编码

### 文件路径

- 所有路径使用相对于项目根目录的相对路径
- 数据集路径可配置，默认在项目根目录下

### 配置管理

- 训练配置在 `train.py` 中修改
- 预测配置在 `yolo.py` 的 `_defaults` 字典中修改
- 评估配置在 `get_map.py` 中修改

### Git 使用

- 不要提交 `logs/` 目录下的权重文件
- 不要提交数据集文件
- 提交前检查 `.gitignore` 配置

## 扩展功能

### COCO 数据集支持

使用 `utils_coco/coco_annotation.py` 处理 COCO 格式数据集：

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

## 参考资源

- 原始仓库: https://github.com/bubbliiiing/yolov5-pytorch
- 常见问题汇总: 见项目根目录 `常见问题汇总.md`
- 相关博客: https://blog.csdn.net/weixin_44791964

## 注意事项

1. **预训练权重**: 对于 99% 的情况都必须使用，不使用会导致训练效果很差
2. **数据集大小**: 建议至少 500 张图片，小数据集需要更长的训练时间
3. **显存管理**: 根据显卡显存调整 `batch_size` 和 `input_shape`
4. **训练时长**: SGD 优化器需要更长的训练时间（300+ epochs），Adam 可以更短（100+ epochs）
5. **评估指标**: mAP 是主要评估指标，Loss 仅用于判断收敛
6. **版本兼容**: 注意 PyTorch、CUDA、cuDNN 版本的兼容性

## 快速开始示例

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

---

**最后更新**: 2025-10-20
