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

def get_fusion_annotation_path(split='train'):
    """
    获取 Fusion 数据集标注文件路径
    
    Args:
        split: 'train', 'val', 或 'test'
    
    Returns:
        Fusion 标注文件的完整路径
    """
    return os.path.join(FUSION_ROOT, 'annotations', f'instances_{split}.json')

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

# 高分辨率模式配置
HIGH_RES = False  # 是否启用高分辨率模式（160x160, 80x80, 40x40）

def configure_high_res_mode(enable_high_res=False, four_features=False):
    """
    配置高分辨率模式
    
    Args:
        enable_high_res: 是否启用高分辨率模式
        four_features: 是否使用四个特征层（P2, P3, P4, P5）
    
    Returns:
        配置后的anchors_path和anchors_mask
    """
    global HIGH_RES, ANCHORS_PATH, ANCHORS_MASK
    
    HIGH_RES = enable_high_res
    
    if HIGH_RES:
        if four_features:
            ANCHORS_PATH = 'model_data/yolo_anchors_four_feat.txt'
            # 四特征层的锚点掩码
            # 160x160使用新增最小锚点[5,6],[8,10],[12,15]
            # 80x80使用原第三组锚点[10,13],[16,30],[33,23]
            # 40x40使用原第二组锚点[30,61],[62,45],[59,119]
            # 20x20使用原第一组锚点[116,90],[156,198],[373,326]
            ANCHORS_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        else:
            # 原来的三特征层高分辨率模式（保留兼容性）
            ANCHORS_PATH = 'model_data/yolo_anchors_high_res.txt'
            ANCHORS_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        
        # 高分辨率模式下建议使用较小的输入尺寸以提高小目标检测能力
        if INPUT_SHAPE[0] > 640:
            print("警告: 高分辨率模式下建议使用较小的输入尺寸(<=640)以提高小目标检测能力")
    else:
        ANCHORS_PATH = 'model_data/yolo_anchors.txt'
        # 标准YOLOv5的锚点掩码
        ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    return ANCHORS_PATH, ANCHORS_MASK

# 初始化配置（默认不启用高分辨率模式）
configure_high_res_mode(HIGH_RES)

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
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300

# Batch Size
FREEZE_BATCH_SIZE = 16
UNFREEZE_BATCH_SIZE = 16  # FRED数据集目标较小，使用较小的batch

# 是否冻结训练
FREEZE_TRAIN = True

# ============================================================================
# 优化器配置
# ============================================================================

# 优化器类型: 'adam' 或 'sgd'
OPTIMIZER_TYPE = 'sgd'

# 学习率
INIT_LR = 2e-2
MIN_LR = INIT_LR * 0.001

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
EVAL_PERIOD = 1  # 调整评估周期：15 -> 5，适配50轮训练

# 数据加载（针对多核CPU优化）
NUM_WORKERS = 8              # CPU核心数的一半，避免过度占用
PREFETCH_FACTOR = 8          # 每个worker预取4个batch，加速数据流
PERSISTENT_WORKERS = True    # 保持workers存活，避免epoch间重复创建进程

# ============================================================================
# Fusion 模型配置
# ============================================================================

# Fusion 训练模态: 'dual', 'rgb', 'event'
# - dual: RGB + Event 双模态融合（默认）
# - rgb: 仅使用 RGB 模态
# - event: 仅使用 Event 模态
FUSION_MODALITY = 'dual'

# Fusion 压缩比率 (0.25 ~ 1.0)
# 控制融合后的特征通道数相对于原始特征的比例
# - 1.0: 不压缩，保留所有特征
# - 0.75: 压缩到75%（推荐）
# - 0.5: 压缩到50%
# - 0.25: 压缩到25%（最大压缩）
FUSION_COMPRESSION_RATIO = 0.75

# Fusion 数据集根目录
FUSION_ROOT = 'datasets/fred_fusion'

# Fusion 训练配置
FUSION_FREEZE_BATCH_SIZE = 8
FUSION_UNFREEZE_BATCH_SIZE = 8

# Fusion 评估配置
FUSION_MAX_EVAL_SAMPLES = 10000  # 最大评估样本数，避免评估时间过长

# Fusion 保存配置
FUSION_SAVE_PERIOD = 10  # 每10个epoch保存一次模型
FUSION_EVAL_PERIOD = 5   # 每5个epoch评估一次mAP

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
        modality: 'rgb', 'event', 或 'fusion'
    
    Returns:
        保存目录路径
    """
    return f'{SAVE_DIR_PREFIX}_{modality}'

def get_model_path(modality='rgb', best=True):
    """
    获取模型权重路径
    
    Args:
        modality: 'rgb', 'event', 或 'fusion'
        best: True返回best权重路径，False返回final权重路径
    
    Returns:
        模型权重路径
    """
    save_dir = get_save_dir(modality)
    if best:
        if modality == 'fusion':
            return os.path.join(save_dir, 'fred_fusion_best.pth')
        else:
            return os.path.join(save_dir, 'best_epoch_weights.pth')
    else:
        if modality == 'fusion':
            return os.path.join(save_dir, 'fred_fusion_final.pth')
        else:
            return os.path.join(save_dir, f'fred_{modality}_final.pth')

def get_fusion_config():
    """
    获取 Fusion 模型配置
    
    Returns:
        Fusion 配置字典
    """
    return {
        'modality': FUSION_MODALITY,
        'compression_ratio': FUSION_COMPRESSION_RATIO,
        'freeze_batch_size': FUSION_FREEZE_BATCH_SIZE,
        'unfreeze_batch_size': FUSION_UNFREEZE_BATCH_SIZE,
        'max_eval_samples': FUSION_MAX_EVAL_SAMPLES,
        'save_period': FUSION_SAVE_PERIOD,
        'eval_period': FUSION_EVAL_PERIOD,
        'save_dir': get_save_dir('fusion'),
        'train_annotation': get_fusion_annotation_path('train'),
        'val_annotation': get_fusion_annotation_path('val'),
        'test_annotation': get_fusion_annotation_path('test'),
        'model_path_best': get_model_path('fusion', best=True),
        'model_path_final': get_model_path('fusion', best=False)
    }

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
    
    # 检查Fusion根目录
    if not os.path.exists(FUSION_ROOT):
        print(f"警告: Fusion数据集根目录不存在: {FUSION_ROOT}")
        print("  如果需要训练Fusion模型，请先运行 convert_fred_to_fusion.py")
    
    # 检查输入尺寸
    if INPUT_SHAPE[0] % 32 != 0 or INPUT_SHAPE[1] % 32 != 0:
        errors.append(f"输入尺寸必须是32的倍数: {INPUT_SHAPE}")
    
    # 检查先验框文件
    if not os.path.exists(ANCHORS_PATH):
        errors.append(f"先验框文件不存在: {ANCHORS_PATH}")
    
    # 检查Fusion压缩比率
    if not (0.25 <= FUSION_COMPRESSION_RATIO <= 1.0):
        errors.append(f"Fusion压缩比率必须在0.25~1.0之间: {FUSION_COMPRESSION_RATIO}")
    
    # 检查Fusion模态
    if FUSION_MODALITY not in ['dual', 'rgb', 'event']:
        errors.append(f"Fusion模态必须是 'dual', 'rgb', 或 'event': {FUSION_MODALITY}")
    
    # 高分辨率模式特定检查
    if HIGH_RES:
        print("高分辨率模式已启用:")
        print(f"  - 使用高分辨率先验框: {ANCHORS_PATH}")
        print(f"  - 锚点掩码: {ANCHORS_MASK}")
        print(f"  - 特征层: 160x160, 80x80, 40x40")
        print(f"  - 输入尺寸: {INPUT_SHAPE}")
        
        # 对于小目标检测，建议使用较小的输入尺寸
        if INPUT_SHAPE[0] > 640:
            errors.append(f"高分辨率模式下建议使用较小的输入尺寸(<=640)，当前为: {INPUT_SHAPE}")
    else:
        print("标准分辨率模式已启用:")
        print(f"  - 使用标准先验框: {ANCHORS_PATH}")
        print(f"  - 锚点掩码: {ANCHORS_MASK}")
        print(f"  - 特征层: 80x80, 40x40, 20x20")
        print(f"  - 输入尺寸: {INPUT_SHAPE}")
    
    # Fusion配置检查
    print("\nFusion配置:")
    print(f"  - 训练模态: {FUSION_MODALITY}")
    print(f"  - 压缩比率: {FUSION_COMPRESSION_RATIO}")
    print(f"  - 冻结批次大小: {FUSION_FREEZE_BATCH_SIZE}")
    print(f"  - 解冻批次大小: {FUSION_UNFREEZE_BATCH_SIZE}")
    print(f"  - 最大评估样本数: {FUSION_MAX_EVAL_SAMPLES}")
    
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

3. 训练单模态：
   # 训练RGB模态
   python train_fred.py --modality rgb
   
   # 训练Event模态
   python train_fred.py --modality event

4. 训练融合模型：
   # 转换融合数据集（确保已运行）
   python convert_fred_to_fusion.py
   
   # 训练融合模型
   python train_fred_fusion.py --modality dual
   
   # 自定义压缩比率
   python train_fred_fusion.py --modality dual --compression_ratio 0.5

5. 自定义配置：
   直接修改本文件中的配置参数

6. 断点续练：
   python train_fred.py --modality rgb --resume
   python train_fred_fusion.py --resume

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
    print(f"Fusion 训练集标注: {get_fusion_annotation_path('train')}")
    print(f"RGB 保存目录: {get_save_dir('rgb')}")
    print(f"Event 保存目录: {get_save_dir('event')}")
    print(f"Fusion 保存目录: {get_save_dir('fusion')}")
    print(f"RGB 最佳模型: {get_model_path('rgb', best=True)}")
    print(f"Event 最终模型: {get_model_path('event', best=False)}")
    print(f"Fusion 最佳模型: {get_model_path('fusion', best=True)}")
    
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
