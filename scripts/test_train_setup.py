#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练脚本的设置（不实际训练）
"""
import os
import json
import torch

def test_dataset_loading(modality='rgb'):
    """测试数据集加载"""
    print(f"\n{'='*70}")
    print(f"测试 {modality.upper()} 数据集加载")
    print(f"{'='*70}\n")
    
    # 检查数据集路径
    coco_root = f'datasets/fred_coco/{modality}'
    train_json = os.path.join(coco_root, 'annotations', 'instances_train.json')
    val_json = os.path.join(coco_root, 'annotations', 'instances_val.json')
    test_json = os.path.join(coco_root, 'annotations', 'instances_test.json')
    
    print("1. 检查数据集文件...")
    for name, path in [('训练集', train_json), ('验证集', val_json), ('测试集', test_json)]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"   ✅ {name}: {len(data['images'])} 张图片, {len(data['annotations'])} 个标注")
        else:
            print(f"   ❌ {name}: 文件不存在 - {path}")
            return False
    
    # 检查图片目录
    print("\n2. 检查图片目录...")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(coco_root, split)
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            print(f"   ✅ {split}: {img_count} 张图片")
        else:
            print(f"   ❌ {split}: 目录不存在 - {img_dir}")
            return False
    
    # 检查CUDA
    print("\n3. 检查CUDA...")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print(f"   ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 检查模型权重目录
    print("\n4. 检查模型目录...")
    model_data_dir = 'model_data'
    if os.path.exists(model_data_dir):
        print(f"   ✅ 模型目录存在: {model_data_dir}")
        anchors_file = os.path.join(model_data_dir, 'yolo_anchors.txt')
        if os.path.exists(anchors_file):
            print(f"   ✅ 先验框文件存在: {anchors_file}")
        else:
            print(f"   ⚠️  先验框文件不存在: {anchors_file}")
    else:
        print(f"   ❌ 模型目录不存在: {model_data_dir}")
        return False
    
    # 检查日志目录
    print("\n5. 检查日志目录...")
    log_dir = f'logs/fred_{modality}'
    os.makedirs(log_dir, exist_ok=True)
    print(f"   ✅ 日志目录: {log_dir}")
    
    print(f"\n{'='*70}")
    print(f"✅ {modality.upper()} 数据集设置检查通过！")
    print(f"{'='*70}\n")
    
    return True

def test_imports():
    """测试必要的导入"""
    print(f"\n{'='*70}")
    print("测试Python包导入")
    print(f"{'='*70}\n")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {e}")
        return False
    
    try:
        from pycocotools.coco import COCO
        print(f"✅ pycocotools")
    except ImportError as e:
        print(f"⚠️  pycocotools导入失败: {e}")
        print(f"   提示: pip install pycocotools")
    
    print(f"\n{'='*70}")
    print("✅ 包导入检查通过！")
    print(f"{'='*70}\n")
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试训练脚本设置')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                       help='选择模态: rgb 或 event')
    args = parser.parse_args()
    
    # 测试导入
    if not test_imports():
        print("\n❌ 包导入测试失败，请安装缺失的包")
        exit(1)
    
    # 测试数据集
    if not test_dataset_loading(args.modality):
        print(f"\n❌ {args.modality.upper()} 数据集测试失败")
        exit(1)
    
    print("\n" + "="*70)
    print("🎉 所有测试通过！可以开始训练")
    print("="*70)
    print(f"\n训练命令:")
    print(f"  python train_fred.py --modality {args.modality}")
    print(f"\n快速训练（不评估mAP）:")
    print(f"  python train_fred.py --modality {args.modality} --no_eval_map")
    print()
