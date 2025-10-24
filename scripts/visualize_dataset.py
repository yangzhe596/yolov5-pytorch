#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化修复后的数据集
验证时间戳对应关系是否正确
"""

import os
import cv2
import json
from pathlib import Path
import argparse


def visualize_coco_sample(coco_root, modality='rgb', split='train', num_samples=5):
    """可视化COCO数据集样本"""
    
    print("=" * 80)
    print(f"可视化COCO数据集 - {modality.upper()} - {split}")
    print("=" * 80)
    
    # 读取COCO标注
    anno_file = Path(coco_root) / modality / 'annotations' / f'instances_{split}.json'
    
    if not anno_file.exists():
        print(f"❌ 标注文件不存在: {anno_file}")
        return
    
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    print(f"\n📊 数据集统计:")
    print(f"   图片数: {len(images)}")
    print(f"   标注数: {len(annotations)}")
    
    # 创建image_id到annotations的映射
    anno_dict = {}
    for anno in annotations:
        img_id = anno['image_id']
        if img_id not in anno_dict:
            anno_dict[img_id] = []
        anno_dict[img_id].append(anno)
    
    # 随机选择样本
    import random
    random.seed(42)
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    print(f"\n🖼️  可视化 {len(sample_images)} 个样本:")
    
    output_dir = Path('visualization_fixed')
    output_dir.mkdir(exist_ok=True)
    
    for i, img_info in enumerate(sample_images):
        img_id = img_info['id']
        filename = img_info['file_name']
        img_path = Path(coco_root) / modality / split / filename
        
        if not img_path.exists():
            print(f"   ⚠️  图片不存在: {img_path}")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        
        # 绘制边界框
        if img_id in anno_dict:
            for anno in anno_dict[img_id]:
                bbox = anno['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 绘制矩形
                cv2.rectangle(img, 
                            (int(x), int(y)), 
                            (int(x + w), int(y + h)), 
                            (0, 255, 0), 2)
                
                # 添加标签
                label = f"ID:{img_id}"
                cv2.putText(img, label, 
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        # 添加时间戳信息
        rel_time = img_info.get('relative_timestamp', 0)
        abs_time = img_info.get('absolute_timestamp', 0)
        
        info_text = f"Rel: {rel_time:.3f}s, Abs: {abs_time:.3f}s"
        cv2.putText(img, info_text,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 0), 2)
        
        # 保存
        output_path = output_dir / f"{modality}_{split}_{i+1}_{filename}"
        cv2.imwrite(str(output_path), img)
        
        print(f"   {i+1}. {filename}")
        print(f"      相对时间: {rel_time:.6f}s")
        print(f"      绝对时间: {abs_time:.6f}s")
        print(f"      标注数: {len(anno_dict.get(img_id, []))}")
        print(f"      保存至: {output_path}")
    
    print(f"\n✅ 可视化完成！输出目录: {output_dir}")
    print("=" * 80)


def compare_with_original(video_id=3, sample_filename=None):
    """对比原始数据和转换后的数据"""
    
    print("\n" + "=" * 80)
    print(f"对比原始数据和转换后的数据")
    print("=" * 80)
    
    # 原始数据路径
    seq_dir = Path(f'/mnt/data/datasets/fred/{video_id}')
    coord_file = seq_dir / 'coordinates.txt'
    rgb_dir = seq_dir / 'PADDED_RGB'
    
    # 读取coordinates.txt
    annotations = {}
    with open(coord_file, 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                ts = float(parts[0])
                coords_str = parts[1].strip()
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    annotations[ts] = coords
    
    # 如果没有指定样本，选择第一个有标注的图片
    if sample_filename is None:
        # 获取所有RGB图片
        rgb_images = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])
        first_image = rgb_images[0]
        
        # 计算第一张图片的绝对时间
        parts = first_image.name.replace('.jpg', '').split('_')
        hours = int(parts[2])
        minutes = int(parts[3])
        seconds = float(parts[4])
        first_abs_time = hours * 3600 + minutes * 60 + seconds
        
        # 找到第一个标注对应的图片
        first_anno_time = min(annotations.keys())
        target_abs_time = first_abs_time + first_anno_time
        
        # 转换为时:分:秒
        hours = int(target_abs_time // 3600)
        minutes = int((target_abs_time % 3600) // 60)
        seconds = target_abs_time % 60
        
        # 查找最接近的图片
        target_pattern = f"Video_{video_id}_{hours:02d}_{minutes:02d}_{seconds:09.6f}"
        
        closest_image = None
        min_diff = float('inf')
        
        for img in rgb_images:
            img_parts = img.name.replace('.jpg', '').split('_')
            img_h = int(img_parts[2])
            img_m = int(img_parts[3])
            img_s = float(img_parts[4])
            img_abs = img_h * 3600 + img_m * 60 + img_s
            
            diff = abs(img_abs - target_abs_time)
            if diff < min_diff:
                min_diff = diff
                closest_image = img.name
        
        sample_filename = closest_image
    
    print(f"\n📋 样本: {sample_filename}")
    
    # 读取原始图片
    img_path = rgb_dir / sample_filename
    img = cv2.imread(str(img_path))
    
    # 提取时间戳
    parts = sample_filename.replace('.jpg', '').split('_')
    hours = int(parts[2])
    minutes = int(parts[3])
    seconds = float(parts[4])
    abs_time = hours * 3600 + minutes * 60 + seconds
    
    # 计算相对时间
    rgb_images = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])
    first_image = rgb_images[0]
    first_parts = first_image.name.replace('.jpg', '').split('_')
    first_abs = int(first_parts[2]) * 3600 + int(first_parts[3]) * 60 + float(first_parts[4])
    
    rel_time = abs_time - first_abs
    
    print(f"   绝对时间: {abs_time:.6f}s ({hours:02d}:{minutes:02d}:{seconds:09.6f})")
    print(f"   相对时间: {rel_time:.6f}s")
    
    # 找到最接近的标注
    closest_anno_time = min(annotations.keys(), key=lambda t: abs(t - rel_time))
    time_diff = abs(closest_anno_time - rel_time)
    bbox = annotations[closest_anno_time]
    
    print(f"   最近标注时间: {closest_anno_time:.6f}s")
    print(f"   时间差: {time_diff:.6f}s")
    print(f"   边界框: {bbox}")
    
    # 绘制边界框
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # 添加信息
    info_text = f"Rel: {rel_time:.3f}s, Anno: {closest_anno_time:.3f}s, Diff: {time_diff:.3f}s"
    cv2.putText(img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 保存
    output_dir = Path('visualization_fixed')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"original_{video_id}_{sample_filename}"
    cv2.imwrite(str(output_path), img)
    
    print(f"   保存至: {output_path}")
    
    if time_diff <= 0.05:
        print(f"\n✅ 时间戳匹配正确！")
    else:
        print(f"\n⚠️  时间戳匹配可能有问题，时间差 {time_diff:.6f}s > 0.05s")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化修复后的数据集')
    parser.add_argument('--coco_root', type=str, default='datasets/fred_coco',
                       help='COCO数据集根目录')
    parser.add_argument('--modality', type=str, default='rgb',
                       choices=['rgb', 'event'],
                       help='模态')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='数据集划分')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='可视化样本数')
    parser.add_argument('--compare_original', action='store_true',
                       help='对比原始数据')
    parser.add_argument('--video_id', type=int, default=3,
                       help='视频序列ID（用于对比原始数据）')
    
    args = parser.parse_args()
    
    if args.compare_original:
        compare_with_original(args.video_id)
    else:
        visualize_coco_sample(args.coco_root, args.modality, args.split, args.num_samples)
