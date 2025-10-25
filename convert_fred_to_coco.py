#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从coordinates.txt重新生成FRED COCO数据集
使用正确的数据源：
- RGB图像：PADDED_RGB文件夹
- Event图像：Event/Frames文件夹
- 标注：coordinates.txt（时间戳: x1, y1, x2, y2）
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil
from tqdm import tqdm


def parse_coordinates_file(coord_file):
    """
    解析coordinates.txt文件
    格式: timestamp: x1, y1, x2, y2
    
    Returns:
        dict: {timestamp: [x1, y1, x2, y2]}
    """
    annotations = {}
    
    with open(coord_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 分割时间戳和坐标
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                
                timestamp = float(parts[0].strip())
                coords_str = parts[1].strip()
                
                # 解析坐标
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    annotations[timestamp] = coords
            except Exception as e:
                print(f"警告: 无法解析行 '{line}': {e}")
                continue
    
    return annotations


def extract_absolute_timestamp_from_filename(filename, modality='rgb'):
    """
    从文件名中提取时间戳（秒）
    
    RGB: Video_0_16_03_03.363444.jpg -> 16*3600 + 03*60 + 03.363444 (绝对时间)
    Event: Video_0_frame_100032333.png -> 100.032333 (相对时间，从Event相机开始录制时计时)
    
    注意：Event的时间戳是相对时间，但起始点与RGB不同
    """
    try:
        if modality == 'rgb':
            # Video_0_16_03_03.363444.jpg
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) >= 5:
                hours = int(parts[2])
                minutes = int(parts[3])
                seconds = float(parts[4])
                timestamp = hours * 3600 + minutes * 60 + seconds
                return timestamp
        else:  # event
            # Video_0_frame_100032333.png
            # 时间戳是微秒数，转换为秒
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 3:
                microseconds = int(parts[-1])
                timestamp = microseconds / 1000000.0
                return timestamp
    except Exception as e:
        print(f"警告: 无法从文件名 '{filename}' 提取时间戳: {e}")
        return None
    
    return None


def find_closest_annotation(relative_timestamp, annotations, tolerance=0.05):
    """
    找到最接近的标注
    
    Args:
        relative_timestamp: 图像相对时间戳（秒）
        annotations: {relative_timestamp: [x1, y1, x2, y2]}
        tolerance: 时间容差（秒）
    
    Returns:
        [x1, y1, x2, y2] or None
    """
    if not annotations:
        return None
    
    # 找到最接近的时间戳
    closest_time = min(annotations.keys(), key=lambda t: abs(t - relative_timestamp))
    
    # 检查是否在容差范围内
    if abs(closest_time - relative_timestamp) <= tolerance:
        return annotations[closest_time]
    
    return None


def calculate_video_start_time(image_files, annotations, modality='rgb'):
    """
    计算视频的起始时间
    
    RGB模态：
    - RGB的第一张图被认为是0时刻
    - coordinates.txt中的时间戳是相对于第一张RGB图片的相对时间
    - video_start_time = 第一张RGB图片的绝对时间
    
    Event模态：
    - Event图像的时间戳（微秒转秒）和coordinates.txt中的时间戳（秒）完全对应
    - 只需要单位转换，不需要计算起始时间
    - 返回0（因为Event时间戳本身就是相对时间）
    
    Args:
        image_files: 图像文件列表
        annotations: {relative_time: bbox}
        modality: 'rgb' or 'event'
    
    Returns:
        start_time: 视频起始时间
    """
    if not image_files:
        return None
    
    if modality == 'event':
        # Event图像的时间戳已经是相对时间（与coordinates.txt对应）
        # 不需要计算起始时间，返回0
        return 0.0
    
    # RGB模态：获取第一张图像的绝对时间
    first_image = sorted(image_files)[0]
    first_image_abs_time = extract_absolute_timestamp_from_filename(first_image.name, modality)
    
    if first_image_abs_time is None:
        return None
    
    # RGB的第一张图被认为是0时刻
    # video_start_time就是第一张图片的绝对时间
    video_start_time = first_image_abs_time
    
    return video_start_time


def convert_fred_to_coco(fred_root, output_root, modality='rgb', 
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                         time_tolerance=0.05):
    """
    转换FRED数据集到COCO格式
    从coordinates.txt读取标注，使用PADDED_RGB和Event/Frames图像
    
    Args:
        fred_root: FRED数据集根目录
        output_root: 输出目录
        modality: 'rgb' 或 'event'
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        time_tolerance: 时间匹配容差（秒）
    """
    try:
        from PIL import Image
    except ImportError:
        print("错误: 需要安装 Pillow 库")
        print("请运行: pip install Pillow")
        return None
    
    fred_root = Path(fred_root)
    output_root = Path(output_root) / modality
    
    # 创建输出目录（仅创建 annotations 目录，不创建 train/val/test）
    (output_root / 'annotations').mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"转换FRED数据集到COCO格式 - {modality.upper()}模态")
    print(f"{'='*70}")
    print(f"源目录: {fred_root}")
    print(f"输出目录: {output_root}")
    print(f"图像源: {'PADDED_RGB' if modality == 'rgb' else 'Event/Frames'}")
    print(f"标注源: coordinates.txt")
    print(f"时间容差: {time_tolerance}秒")
    print(f"数据集划分: 训练{train_ratio*100:.0f}% / 验证{val_ratio*100:.0f}% / 测试{test_ratio*100:.0f}%")
    print(f"模式: 使用相对路径（不复制图像文件）")
    print(f"{'='*70}\n")
    
    # COCO格式数据结构
    coco_data = {
        'train': {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'object', 'supercategory': 'object'}]
        },
        'val': {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'object', 'supercategory': 'object'}]
        },
        'test': {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'object', 'supercategory': 'object'}]
        }
    }
    
    image_id = 1
    annotation_id = 1
    
    # 统计信息
    stats = {
        'total_sequences': 0,
        'total_images': 0,
        'matched_images': 0,
        'unmatched_images': 0,
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'total_annotations': 0
    }
    
    # 遍历所有序列
    sequences = sorted([d for d in fred_root.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"找到 {len(sequences)} 个序列\n")
    
    for seq_dir in tqdm(sequences, desc="处理序列"):
        stats['total_sequences'] += 1
        
        # 查找coordinates.txt
        coord_file = seq_dir / 'coordinates.txt'
        if not coord_file.exists():
            # 尝试子目录（处理序列2的特殊情况）
            subdirs = [d for d in seq_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for subdir in subdirs:
                coord_file = subdir / 'coordinates.txt'
                if coord_file.exists():
                    seq_dir = subdir
                    break
        
        if not coord_file.exists():
            continue
        
        # 解析标注（相对时间戳）
        annotations = parse_coordinates_file(coord_file)
        if not annotations:
            continue
        
        # 确定图像目录
        if modality == 'rgb':
            img_dir = seq_dir / 'PADDED_RGB'
            img_ext = '.jpg'
        else:  # event
            img_dir = seq_dir / 'Event' / 'Frames'
            img_ext = '.png'
        
        if not img_dir.exists():
            continue
        
        # 获取所有图像
        images = sorted([f for f in img_dir.iterdir() if f.suffix == img_ext])
        
        if not images:
            continue
        
        # 计算视频起始时间
        video_start_time = calculate_video_start_time(images, annotations, modality)
        if video_start_time is None:
            continue
        
        # 处理每张图像
        for img_path in images:
            stats['total_images'] += 1
            
            # 提取绝对时间戳
            abs_timestamp = extract_absolute_timestamp_from_filename(img_path.name, modality)
            if abs_timestamp is None:
                stats['unmatched_images'] += 1
                continue
            
            # 转换为相对时间戳
            relative_timestamp = abs_timestamp - video_start_time
            
            # 查找匹配的标注
            bbox = find_closest_annotation(relative_timestamp, annotations, time_tolerance)
            if bbox is None:
                stats['unmatched_images'] += 1
                continue
            
            stats['matched_images'] += 1
            
            # 确定数据集划分（使用确定性方法）
            rand_val = (image_id % 100) / 100.0
            if rand_val < train_ratio:
                split = 'train'
                stats['train_images'] += 1
            elif rand_val < train_ratio + val_ratio:
                split = 'val'
                stats['val_images'] += 1
            else:
                split = 'test'
                stats['test_images'] += 1
            
            # 生成相对路径（相对于 fred_root）
            # 格式: "序列0/PADDED_RGB/Video_0_16_03_03.363444.jpg"
            seq_name = seq_dir.parent.name if seq_dir.parent.name.isdigit() else seq_dir.name
            try:
                relative_path = img_path.relative_to(fred_root)
                new_filename = str(relative_path)
            except ValueError:
                # 如果无法计算相对路径，使用原来的方式
                new_filename = f"seq{seq_name}_{img_path.name}"
            
            # 不复制图像，直接使用原始路径
            
            # 获取图像尺寸
            with Image.open(img_path) as img:
                width, height = img.size
            
            # 添加图像信息
            image_info = {
                'id': image_id,
                'file_name': new_filename,
                'width': width,
                'height': height,
                'sequence': seq_name,
                'relative_timestamp': relative_timestamp,
                'absolute_timestamp': abs_timestamp
            }
            coco_data[split]['images'].append(image_info)
            
            # 转换bbox格式: [x1, y1, x2, y2] -> [x, y, width, height]
            x1, y1, x2, y2 = bbox
            bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            area = float((x2 - x1) * (y2 - y1))
            
            # 添加标注信息
            annotation_info = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox_coco,
                'area': area,
                'iscrowd': 0
            }
            coco_data[split]['annotations'].append(annotation_info)
            
            stats['total_annotations'] += 1
            image_id += 1
            annotation_id += 1
    
    # 保存COCO格式标注文件
    print(f"\n保存COCO标注文件...")
    for split in ['train', 'val', 'test']:
        output_file = output_root / 'annotations' / f'instances_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data[split], f, indent=2)
        print(f"  {split}: {output_file}")
    
    # 打印统计信息
    print(f"\n{'='*70}")
    print(f"转换完成！")
    print(f"{'='*70}")
    print(f"总序列数: {stats['total_sequences']}")
    print(f"总图像数: {stats['total_images']}")
    print(f"匹配图像数: {stats['matched_images']}")
    print(f"未匹配图像数: {stats['unmatched_images']}")
    if stats['total_images'] > 0:
        print(f"匹配率: {stats['matched_images']/stats['total_images']*100:.2f}%")
    print(f"\n数据集划分:")
    print(f"  训练集: {stats['train_images']} 张图像")
    print(f"  验证集: {stats['val_images']} 张图像")
    print(f"  测试集: {stats['test_images']} 张图像")
    print(f"  总标注数: {stats['total_annotations']}")
    print(f"{'='*70}\n")
    
    # 保存统计信息
    stats_file = output_root / 'conversion_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存: {stats_file}\n")
    
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从coordinates.txt重新生成FRED COCO数据集')
    parser.add_argument('--fred_root', type=str, default='/mnt/data/datasets/fred',
                        help='FRED数据集根目录')
    parser.add_argument('--output_root', type=str, default='datasets/fred_coco',
                        help='输出目录')
    parser.add_argument('--modality', type=str, default='both', choices=['rgb', 'event', 'both'],
                        help='转换的模态')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--time_tolerance', type=float, default=0.05,
                        help='时间匹配容差（秒）')
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"数据集比例之和必须为1.0，当前为{total_ratio}")
    
    # 转换数据集
    if args.modality in ['rgb', 'both']:
        print("\n" + "="*70)
        print("转换RGB数据集")
        print("="*70)
        convert_fred_to_coco(
            args.fred_root,
            args.output_root,
            modality='rgb',
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            time_tolerance=args.time_tolerance
        )
    
    if args.modality in ['event', 'both']:
        print("\n" + "="*70)
        print("转换Event数据集")
        print("="*70)
        convert_fred_to_coco(
            args.fred_root,
            args.output_root,
            modality='event',
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            time_tolerance=args.time_tolerance
        )
    
    print("\n✅ 所有转换完成！")
