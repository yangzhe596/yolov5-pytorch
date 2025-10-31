#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED数据集训练统一配置文件
集中管理所有FRED训练相关的配置参数

版本: 2.0
更新: 2025-11-01
- 支持新版 convert_fred_to_coco_v2.py
- 支持帧级别和序列级别划分
- 改进的路径管理
"""
import os

# ============================================================================
# 路径配置
# ============================================================================

# FRED 数据集根目录（可通过环境变量覆盖）
FRED_ROOT = os.environ.get('FRED_ROOT', '/mnt/data/datasets/fred')

# COCO 格式数据集根目录
COCO_ROOT = 'datasets/fred_coco'

# 数据集划分模式: 'frame' 或 'sequence'
# - frame: 帧级别划分（推荐，数据分布更均衡）
# - sequence: 序列级别划分（严格场景分离）
SPLIT_MODE = 'frame'

def get_fred_root():
    """获取 FRED 数据集根目录"""
    return FRED_ROOT

def get_coco_root():
    """获取 COCO 数据集根目录"""
    return COCO_ROOT

def get_image_dir(modality='rgb'):
    """
    获取图像目录
    
    Args:
        modality: 'rgb' 或 'event'
    
    Returns:
        FRED 数据集根目录（因为 file_name 包含相对路径）
    """
    return FRED_ROOT

def get_annotation_path(modality='rgb', split='train'):
    """
    获取标注文件路径
    
    Args:
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
    
    Returns:
        标注文件的完整路径
    """
    return os.path.join(COCO_ROOT, modality, 'annotations', f'instances_{split}.json')

def get_coco_modality_root(modality='rgb'):
    """
    获取指定模态的COCO数据集根目录
    
    Args:
        modality: 'rgb' 或 'event'
    
    Returns:
        COCO数据集模态根目录
    """
    return os.path.join(COCO_ROOT, modality)

# ============================================================================
# 数据集配置
# ============================================================================

# 类别配置（FRED数据集只有一个类别）
NUM_CLASSES = 1
CLASS_NAMES = ['object']

# ============================================================================
# 模型配置
# ============================================================================

# 输入图片尺寸（必须是32的倍数）
INPUT_SHAPE = [640, 640]

# 主干网络: 'cspdarknet', 'convnext_tiny', 'convnext_small', 'swin_transfomer_tiny'
BACKBONE = 'cspdarknet'

# YOLOv5版本: 's', 'm', 'l', 'x'
PHI = 'm'

# 是否使用预训练权重
PRETRAINED = True

# 先验框配置
ANCHORS_PATH = 'model_data/yolo_anchors.txt'
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# ============================================================================
# 训练配置
# ============================================================================

# CUDA配置
CUDA = True
DISTRIBUTED = False
SYNC_BN = False
FP16 = False

# 随机种子
SEED = 11

# 训练轮次
INIT_EPOCH = 0
FREEZE_EPOCH = 15
UNFREEZE_EPOCH = 50

# Batch Size
FREEZE_BATCH_SIZE = 16
UNFREEZE_BATCH_SIZE = 8  # FRED数据集目标较小，使用较小的batch

# 是否冻结训练
FREEZE_TRAIN = True

# ============================================================================
# 优化器配置
# ============================================================================

# 优化器类型: 'adam' 或 'sgd'
OPTIMIZER_TYPE = 'sgd'

# 学习率
INIT_LR = 5e-3
MIN_LR = INIT_LR * 0.01

# SGD参数
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4

# 学习率衰减类型: 'step' 或 'cos'
LR_DECAY_TYPE = 'cos'

# ============================================================================
# 数据增强配置
# ============================================================================

# Mosaic数据增强
MOSAIC = True
MOSAIC_PROB = 0.5

# MixUp数据增强
MIXUP = True
MIXUP_PROB = 0.5

# 特殊数据增强比例（前70%的epoch使用强数据增强）
SPECIAL_AUG_RATIO = 0.7

# 标签平滑
LABEL_SMOOTHING = 0

# ============================================================================
# 训练控制
# ============================================================================

# 保存周期（每多少个epoch保存一次）
SAVE_PERIOD = 5  # 调整保存周期：15 -> 5，适配50轮训练

# 保存目录前缀
SAVE_DIR_PREFIX = 'logs/fred'

# 评估配置
EVAL_FLAG = True
EVAL_PERIOD = 5  # 调整评估周期：15 -> 5，适配50轮训练

# 数据加载（针对多核CPU优化）
NUM_WORKERS = 8  # 增加：4 -> 8，加速数据加载

# ============================================================================
# 评估配置
# ============================================================================

# mAP计算配置
CONFIDENCE = 0.05
NMS_IOU = 0.5
MINOVERLAP = 0.5
MAX_BOXES = 100
LETTERBOX_IMAGE = True

# ============================================================================
# 辅助函数
# ============================================================================

def get_save_dir(modality='rgb'):
    """
    获取保存目录
    
    Args:
        modality: 'rgb' 或 'event'
    
    Returns:
        保存目录路径
    """
    return f'{SAVE_DIR_PREFIX}_{modality}'

def get_model_path(modality='rgb', best=True):
    """
    获取模型权重路径
    
    Args:
        modality: 'rgb' 或 'event'
        best: True返回best权重路径，False返回final权重路径
    
    Returns:
        模型权重路径
    """
    save_dir = get_save_dir(modality)
    if best:
        return os.path.join(save_dir, 'best_epoch_weights.pth')
    else:
        return os.path.join(save_dir, f'fred_{modality}_final.pth')

# ============================================================================
# 配置验证
# ============================================================================

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查FRED根目录
    if not os.path.exists(FRED_ROOT):
        errors.append(f"FRED根目录不存在: {FRED_ROOT}")
    
    # 检查COCO根目录
    if not os.path.exists(COCO_ROOT):
        errors.append(f"COCO数据集根目录不存在: {COCO_ROOT}")
    
    # 检查输入尺寸
    if INPUT_SHAPE[0] % 32 != 0 or INPUT_SHAPE[1] % 32 != 0:
        errors.append(f"输入尺寸必须是32的倍数: {INPUT_SHAPE}")
    
    # 检查先验框文件
    if not os.path.exists(ANCHORS_PATH):
        errors.append(f"先验框文件不存在: {ANCHORS_PATH}")
    
    return errors

# ============================================================================
# 使用说明
# ============================================================================

"""
快速开始：

1. 设置FRED数据集根目录（可选）：
   export FRED_ROOT=/path/to/your/fred/dataset

2. 转换FRED数据集为COCO格式（使用新版本脚本）：
   # 帧级别划分（推荐）
   python convert_fred_to_coco_v2.py --split-mode frame --modality both
   
   # 序列级别划分
   python convert_fred_to_coco_v2.py --split-mode sequence --modality both

3. 训练RGB模态：
   python train_fred.py --modality rgb

4. 训练Event模态：
   python train_fred.py --modality event

5. 自定义配置：
   直接修改本文件中的配置参数

6. 断点续练：
   python train_fred.py --modality rgb --resume

配置优先级：
1. 命令行参数（最高优先级）
2. 本配置文件
3. 默认值（最低优先级）

新版本改进：
- 使用 interpolated_coordinates.txt（包含 drone_id）
- 支持帧级别和序列级别划分
- 完整的数据验证
- 详细的统计信息

详细文档：
- FRED_DATASET_CONVERSION_GUIDE.md - 数据集转换指南
- AGENTS.md - 完整项目文档
"""

if __name__ == '__main__':
    # 测试配置
    print("=" * 70)
    print("FRED 数据集配置")
    print("=" * 70)
    print(f"FRED 根目录: {get_fred_root()}")
    print(f"COCO 根目录: {get_coco_root()}")
    print(f"RGB 图像目录: {get_image_dir('rgb')}")
    print(f"Event 图像目录: {get_image_dir('event')}")
    print(f"RGB 训练集标注: {get_annotation_path('rgb', 'train')}")
    print(f"Event 测试集标注: {get_annotation_path('event', 'test')}")
    print(f"RGB 保存目录: {get_save_dir('rgb')}")
    print(f"Event 保存目录: {get_save_dir('event')}")
    print(f"RGB 最佳模型: {get_model_path('rgb', best=True)}")
    print(f"Event 最终模型: {get_model_path('event', best=False)}")
    
    # 验证配置
    print("\n" + "=" * 70)
    print("配置验证")
    print("=" * 70)
    errors = validate_config()
    if errors:
        print("❌ 发现以下配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ 配置验证通过")
