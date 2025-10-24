#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化COCO数据集的样本，检查边界框是否正确
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import random


def visualize_coco_samples(coco_root, modality='rgb', split='train', num_samples=10, output_dir='coco_visualization'):
    """
    可视化COCO数据集的样本
    
    Args:
        coco_root: COCO数据集根目录
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
        num_samples: 可视化的样本数量
        output_dir: 输出目录
    """
    coco_root = Path(coco_root) / modality
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO标注文件
    ann_file = coco_root / 'annotations' / f'instances_{split}.json'
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"可视化COCO数据集 - {modality.upper()} {split}")
    print(f"{'='*70}")
    print(f"标注文件: {ann_file}")
    print(f"图像数量: {len(coco_data['images'])}")
    print(f"标注数量: {len(coco_data['annotations'])}")
    print(f"{'='*70}\n")
    
    # 创建image_id到annotations的映射
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # 随机选择样本
    images = coco_data['images']
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    
    # 可视化每个样本
    for img_info in images:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = coco_root / split / img_name
        
        if not img_path.exists():
            print(f"⚠️  图片不存在: {img_path}")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  无法读取图片: {img_path}")
            continue
        
        # 获取标注
        anns = img_id_to_anns.get(img_id, [])
        
        # 绘制边界框
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 转换为整数
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"ID:{ann['id']}"
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加图片信息
        info_text = f"{img_name} | {len(anns)} objects"
        cv2.putText(img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存可视化结果
        output_path = output_dir / f"{modality}_{split}_{img_name}"
        cv2.imwrite(str(output_path), img)
        print(f"✅ 已保存: {output_path}")
    
    print(f"\n{'='*70}")
    print(f"可视化完成！共处理 {len(images)} 张图片")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")


def check_bbox_validity(coco_root, modality='rgb', split='train'):
    """
    检查边界框的有效性
    """
    coco_root = Path(coco_root) / modality
    
    # 读取COCO标注文件
    ann_file = coco_root / 'annotations' / f'instances_{split}.json'
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"检查边界框有效性 - {modality.upper()} {split}")
    print(f"{'='*70}")
    
    # 创建image_id到image_info的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 统计信息
    stats = {
        'total': 0,
        'valid': 0,
        'out_of_bounds': 0,
        'zero_area': 0,
        'negative_coords': 0,
        'too_small': 0,
        'too_large': 0
    }
    
    issues = []
    
    for ann in coco_data['annotations']:
        stats['total'] += 1
        
        img_info = img_id_to_info[ann['image_id']]
        img_width = img_info['width']
        img_height = img_info['height']
        
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # 检查各种问题
        has_issue = False
        
        # 负坐标
        if x < 0 or y < 0:
            stats['negative_coords'] += 1
            has_issue = True
        
        # 零面积
        if w <= 0 or h <= 0:
            stats['zero_area'] += 1
            has_issue = True
        
        # 超出边界
        if x + w > img_width or y + h > img_height:
            stats['out_of_bounds'] += 1
            has_issue = True
        
        # 太小（可能是标注错误）
        if w < 5 or h < 5:
            stats['too_small'] += 1
            has_issue = True
        
        # 太大（可能是标注错误）
        if w > img_width * 0.9 or h > img_height * 0.9:
            stats['too_large'] += 1
            has_issue = True
        
        if not has_issue:
            stats['valid'] += 1
        else:
            issues.append({
                'image': img_info['file_name'],
                'bbox': bbox,
                'img_size': (img_width, img_height),
                'annotation_id': ann['id']
            })
    
    # 打印统计信息
    print(f"总标注数: {stats['total']}")
    print(f"有效标注: {stats['valid']} ({stats['valid']/stats['total']*100:.2f}%)")
    print(f"\n问题统计:")
    print(f"  负坐标: {stats['negative_coords']}")
    print(f"  零面积: {stats['zero_area']}")
    print(f"  超出边界: {stats['out_of_bounds']}")
    print(f"  太小(<5px): {stats['too_small']}")
    print(f"  太大(>90%): {stats['too_large']}")
    
    # 显示一些问题样本
    if issues:
        print(f"\n前10个问题样本:")
        for i, issue in enumerate(issues[:10], 1):
            print(f"{i}. {issue['image']}")
            print(f"   bbox: {issue['bbox']}")
            print(f"   img_size: {issue['img_size']}")
    
    print(f"{'='*70}\n")
    
    return stats, issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化COCO数据集样本')
    parser.add_argument('--coco_root', type=str, default='datasets/fred_coco',
                        help='COCO数据集根目录')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='模态')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='数据集划分')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='可视化的样本数量')
    parser.add_argument('--output_dir', type=str, default='coco_visualization',
                        help='输出目录')
    parser.add_argument('--check_only', action='store_true',
                        help='仅检查边界框有效性，不可视化')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_bbox_validity(args.coco_root, args.modality, args.split)
    else:
        # 先检查
        check_bbox_validity(args.coco_root, args.modality, args.split)
        # 再可视化
        visualize_coco_samples(args.coco_root, args.modality, args.split, 
                              args.num_samples, args.output_dir)
