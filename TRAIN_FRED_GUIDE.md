# FRED数据集训练指南

## 快速开始

### 1. 数据准备（已完成）

FRED数据集已转换为COCO格式：
- ✅ RGB模态: `datasets/fred_coco/rgb/` (19,471张图片)
- ✅ Event模态: `datasets/fred_coco/event/` (28,714张图片)

### 2. 训练模型

#### 训练RGB模态

```bash
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb
```

#### 训练Event模态

```bash
/home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality event
```

### 3. 预测测试

```bash
# 在测试集上预测（RGB模态）
/home/yz/miniforge3/envs/torch/bin/python3 predict_fred.py --modality rgb --split test --num_samples 10

# 在测试集上预测（Event模态）
/home/yz/miniforge3/envs/torch/bin/python3 predict_fred.py --modality event --split test --num_samples 10
```

## 详细配置

### 修改训练参数

编辑 `config_fred.py` 文件：

```python
# 选择模态
MODALITY = 'rgb'  # 或 'event'

# 模型配置
INPUT_SHAPE = [640, 640]  # 输入尺寸
BACKBONE = 'cspdarknet'    # 主干网络
PHI = 's'                  # 模型版本

# 训练参数
FREEZE_EPOCH = 50          # 冻结训练轮次
UNFREEZE_EPOCH = 300       # 总训练轮次
FREEZE_BATCH_SIZE = 16     # 冻结阶段batch size
UNFREEZE_BATCH_SIZE = 8    # 解冻阶段batch size

# 优化器
OPTIMIZER_TYPE = 'sgd'     # 优化器类型
INIT_LR = 1e-2             # 初始学习率
```

### 断点续练

```bash
# 修改train_fred.py中的model_path
# 或在命令行中指定（需要添加参数支持）

# 示例：从第60轮继续训练
# 在train_fred.py中设置：
# model_path = 'logs/fred_rgb/ep060-loss0.123.pth'
# Init_Epoch = 60
```

## 文件结构

```
yolov5-pytorch/
├── train_fred.py              # FRED数据集训练脚本
├── predict_fred.py            # FRED数据集预测脚本
├── config_fred.py             # FRED训练配置
├── utils/
│   └── dataloader_coco.py    # COCO格式数据加载器
├── model_data/
│   └── fred_classes.txt      # FRED类别文件
└── datasets/
    └── fred_coco/            # FRED COCO数据集
        ├── rgb/              # RGB模态
        └── event/            # Event模态
```

## 训练输出

训练过程中会生成：

```
logs/fred_{modality}/
├── loss_{timestamp}/         # TensorBoard日志
├── best_epoch_weights.pth    # 最佳权重
├── last_epoch_weights.pth    # 最后一轮权重
└── ep{epoch}-loss{loss}.pth  # 定期保存的权重
```

## 数据集特点

### RGB模态
- 图片数量: 19,471张
- 图片格式: JPG
- 平均目标尺寸: 50.22 × 34.08 像素
- 特点: 小目标检测

### Event模态
- 图片数量: 28,714张
- 图片格式: PNG
- 平均目标尺寸: 50.96 × 34.58 像素
- 特点: 小目标检测，约3%边界框被裁剪

## 训练建议

### 针对小目标优化

1. **调整输入尺寸**
   ```python
   INPUT_SHAPE = [1280, 1280]  # 更大的输入尺寸有助于小目标检测
   ```

2. **调整先验框**
   ```bash
   # 使用kmeans重新计算适合FRED数据集的先验框
   python kmeans_for_anchors.py
   ```

3. **增加训练轮次**
   ```python
   UNFREEZE_EPOCH = 500  # 小目标需要更长的训练时间
   ```

4. **使用更强的数据增强**
   ```python
   MOSAIC = True
   MOSAIC_PROB = 0.7  # 提高Mosaic概率
   ```

### 针对不同模态

#### RGB模态
- 使用标准的数据增强
- 注意颜色抖动参数
- 可以使用预训练权重

#### Event模态
- Event图像可能需要不同的增强策略
- 考虑调整色域变换参数
- 可能需要从头训练或使用RGB预训练权重微调

## 常见问题

### Q1: 显存不足
**解决**:
- 减小batch size: `UNFREEZE_BATCH_SIZE = 4`
- 减小输入尺寸: `INPUT_SHAPE = [416, 416]`
- 使用更小的模型: `PHI = 's'`

### Q2: 训练速度慢
**解决**:
- 增加num_workers: `NUM_WORKERS = 8`
- 使用混合精度训练: `FP16 = True`
- 减小输入尺寸

### Q3: 模型效果不好
**解决**:
- 增加训练轮次
- 调整学习率
- 使用更强的数据增强
- 重新计算先验框
- 尝试不同的主干网络

### Q4: 如何评估模型？
**解决**:
目前COCO格式的mAP评估功能待实现。临时方案：
1. 使用predict_fred.py在测试集上预测
2. 手动检查预测结果
3. 或实现COCO格式的mAP计算

## 测试脚本

### 测试数据加载器

```bash
/home/yz/miniforge3/envs/torch/bin/python3 -c "
from utils.dataloader_coco import CocoYoloDataset
from utils.utils import get_anchors
import numpy as np

# 加载数据集
anchors, _ = get_anchors('model_data/yolo_anchors.txt')
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

dataset = CocoYoloDataset(
    coco_json_path='datasets/fred_coco/rgb/annotations/instances_train.json',
    image_dir='datasets/fred_coco/rgb/train',
    input_shape=[640, 640],
    num_classes=1,
    anchors=anchors,
    anchors_mask=anchors_mask,
    epoch_length=100,
    mosaic=False,
    mixup=False,
    mosaic_prob=0,
    mixup_prob=0,
    train=False,
    special_aug_ratio=0
)

print(f'数据集大小: {len(dataset)}')

# 测试加载一个样本
image, box, y_true = dataset[0]
print(f'图片shape: {image.shape}')
print(f'边界框数量: {len(box)}')
print(f'Y_true层数: {len(y_true)}')
"
```

### 测试训练一个epoch

```bash
# 快速测试（1个epoch）
/home/yz/miniforge3/envs/torch/bin/python3 -c "
import sys
sys.path.insert(0, '.')

# 修改train_fred.py中的UnFreeze_Epoch = 1
# 然后运行
"
```

## 监控训练

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir=logs/fred_rgb/loss_*

# 在浏览器中打开
# http://localhost:6006
```

### 查看日志

```bash
# 查看最新的训练日志
tail -f logs/fred_rgb/loss_*/train.log
```

## 下一步

1. ✅ 数据集已准备好
2. 🔄 开始训练
3. 📋 评估模型性能
4. 📋 优化超参数
5. 📋 部署模型

## 参考

- `AGENTS.md` - 项目整体指南
- `README_FRED_COCO.md` - FRED数据集详细文档
- `QUICK_START_FRED.md` - 快速开始指南
- `config_fred.py` - 训练配置文件

---

**Python环境**: `/home/yz/miniforge3/envs/torch/bin/python3`  
**创建时间**: 2025-10-20
