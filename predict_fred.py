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

def predict_fred_dataset(modality='rgb', split='test', num_samples=10, save_results=True):
    """
    在FRED数据集上进行预测
    
    Args:
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
        num_samples: 预测的样本数量（0表示全部）
        save_results: 是否保存预测结果
    """
    import json
    from pathlib import Path
    
    print("=" * 80)
    print(f"FRED数据集预测 - {modality.upper()}模态 - {split}集")
    print("=" * 80)
    
    # 配置YOLO模型
    # 注意：需要修改yolo.py中的_defaults以匹配FRED模型
    yolo = YOLO(
        model_path=f'logs/fred_{modality}/best_epoch_weights.pth',
        classes_path='model_data/fred_classes.txt',  # 需要创建这个文件
        anchors_path='model_data/yolo_anchors.txt',
        input_shape=[640, 640],
        backbone='cspdarknet',
        phi='s',
        confidence=0.5,
        nms_iou=0.3,
        cuda=True
    )
    
    # 加载COCO标注
    coco_root = f'datasets/fred_coco/{modality}'
    ann_file = f'{coco_root}/annotations/instances_{split}.json'
    # 使用FRED数据集根目录（file_name包含相对路径）
    fred_root = '/home/yz/datasets/fred'
    img_dir = fred_root
    
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
            output_path = os.path.join(output_dir, img_info['file_name'])
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
                        help='模型权重路径（覆盖默认路径）')
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    default_model = f'logs/fred_{args.modality}/best_epoch_weights.pth'
    model_path = args.model_path if args.model_path else default_model
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print(f"\n请先训练模型:")
        print(f"  python train_fred.py --modality {args.modality}")
        return
    
    # 预测
    predict_fred_dataset(
        modality=args.modality,
        split=args.split,
        num_samples=args.num_samples,
        save_results=not args.no_save
    )

if __name__ == "__main__":
    main()
