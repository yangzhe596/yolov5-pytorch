#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用FRED数据集训练的模型进行预测
"""
import argparse
import os
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

# 导入FRED配置
import config_fred as cfg


def predict_fred_dataset(modality='rgb', split='test', num_samples=10, 
                         save_results=True, model_path=None):
    """
    在FRED数据集上进行预测
    
    Args:
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
        num_samples: 预测的样本数量（0表示全部）
        save_results: 是否保存预测结果
        model_path: 模型权重路径（None则使用配置文件中的最佳权重）
    """
    import json
    from pathlib import Path
    
    print("=" * 80)
    print(f"FRED数据集预测 - {modality.upper()}模态 - {split}集")
    print("=" * 80)
    
    # 如果未指定模型路径，使用配置文件中的最佳权重
    if model_path is None:
        model_path = cfg.get_model_path(modality, best=True)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print(f"\n请先训练模型:")
        print(f"  python train_fred.py --modality {modality}")
        return
    
    # 配置YOLO模型（使用配置文件中的参数）
    # 注意：需要创建 model_data/fred_classes.txt 文件
    classes_file = 'model_data/fred_classes.txt'
    if not os.path.exists(classes_file):
        # 自动创建类别文件
        os.makedirs('model_data', exist_ok=True)
        with open(classes_file, 'w') as f:
            for class_name in cfg.CLASS_NAMES:
                f.write(f"{class_name}\n")
        print(f"✓ 已创建类别文件: {classes_file}")
    
    # 使用字典方式传递参数给 YOLO 类
    yolo = YOLO(**{
        'model_path': model_path,
        'classes_path': classes_file,
        'anchors_path': cfg.ANCHORS_PATH,
        'input_shape': cfg.INPUT_SHAPE,
        'backbone': cfg.BACKBONE,
        'phi': cfg.PHI,
        'confidence': 0.5,  # 预测时使用较高的置信度
        'nms_iou': 0.3,
        'cuda': cfg.CUDA
    })
    
    # 加载COCO标注
    ann_file = cfg.get_annotation_path(modality, split)
    img_dir = cfg.get_image_dir(modality)
    
    if not os.path.exists(ann_file):
        print(f"错误: 标注文件不存在 {ann_file}")
        return
    
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    
    # 选择样本
    if num_samples > 0:
        import random
        random.seed(42)
        images = random.sample(images, min(num_samples, len(images)))
    
    print(f"\n预测 {len(images)} 张图片...")
    
    # 创建输出目录
    if save_results:
        output_dir = f'predictions_fred_{modality}_{split}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存到: {output_dir}")
    
    # 预测
    total_time = 0
    for i, img_info in enumerate(images):
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在 {img_path}")
            continue
        
        # 加载图片
        image = Image.open(img_path)
        
        # 预测
        start_time = time.time()
        r_image = yolo.detect_image(image, crop=False, count=True)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # 保存结果
        if save_results:
            # 只使用文件名（不包含目录），避免路径问题
            output_filename = os.path.basename(img_info['file_name'])
            output_path = os.path.join(output_dir, output_filename)
            r_image.save(output_path)
        
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i+1}/{len(images)}, 平均耗时: {total_time/(i+1):.3f}s/张")
    
    # 统计
    avg_time = total_time / len(images) if len(images) > 0 else 0
    fps = 1 / avg_time if avg_time > 0 else 0
    
    print(f"\n预测完成！")
    print(f"  总图片数: {len(images)}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均耗时: {avg_time:.3f}s/张")
    print(f"  FPS: {fps:.2f}")
    
    if save_results:
        print(f"  结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='使用FRED数据集训练的模型进行预测')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='选择模态')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='选择数据集划分')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='预测样本数量（0表示全部）')
    parser.add_argument('--no_save', action='store_true',
                        help='不保存预测结果')
    parser.add_argument('--model_path', type=str, default='',
                        help='模型权重路径（默认使用配置文件中的最佳权重）')
    
    args = parser.parse_args()
    
    # 如果未指定模型路径，使用None（函数内部会使用配置文件中的路径）
    model_path = args.model_path if args.model_path else None
    
    # 预测
    predict_fred_dataset(
        modality=args.modality,
        split=args.split,
        num_samples=args.num_samples,
        save_results=not args.no_save,
        model_path=model_path
    )


if __name__ == "__main__":
    main()
