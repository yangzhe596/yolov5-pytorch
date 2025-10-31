#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化特定图片的标注
支持通过图片ID、文件名、序列号等方式指定图片
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import re


def visualize_specific_images(coco_root='/mnt/data/datasets/fred',
                              output_root='datasets/fred_coco',
                              modality='rgb',
                              split='train',
                              image_ids=None,
                              filenames=None,
                              sequences=None,
                              pattern=None,
                              output_dir='specific_visualization',
                              show_info=True):
    """
    可视化特定图片
    
    Args:
        coco_root: FRED原始数据集根目录
        output_root: COCO数据集根目录
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
        image_ids: 图片ID列表，如 [1, 2, 3]
        filenames: 文件名列表（支持部分匹配），如 ['Video_0_16_03_03', 'Video_1_16_05_12']
        sequences: 序列号列表，如 [1, 3, 5]
        pattern: 文件名正则表达式模式
        output_dir: 输出目录
        show_info: 是否显示详细信息
    """
    
    # 路径设置
    coco_root_path = Path(output_root) / modality
    output_dir_path = Path(output_dir) / f"{modality}_{split}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO标注文件
    ann_file = coco_root_path / 'annotations' / f'instances_{split}.json'
    
    if not ann_file.exists():
        print(f"❌ 标注文件不存在: {ann_file}")
        return None
    
    print(f"\n{'='*80}")
    print(f"🖼️  可视化特定图片 - {modality.upper()} {split.upper()}")
    print(f"{'='*80}")
    print(f"标注文件: {ann_file}")
    print(f"输出目录: {output_dir_path}")
    print(f"{'='*80}\n")
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # 创建image_id到annotations的映射
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # 筛选要可视化的图片
    selected_images = []
    
    # 方式1: 通过图片ID筛选
    if image_ids:
        print(f"📌 通过图片ID筛选: {image_ids}")
        image_ids_set = set(image_ids)
        selected_images = [img for img in images if img['id'] in image_ids_set]
        print(f"   找到 {len(selected_images)} 张图片\n")
    
    # 方式2: 通过文件名筛选（支持部分匹配）
    elif filenames:
        print(f"📌 通过文件名筛选: {filenames}")
        for filename_pattern in filenames:
            matched = [img for img in images if filename_pattern in img['file_name']]
            selected_images.extend(matched)
            print(f"   '{filename_pattern}' 匹配到 {len(matched)} 张图片")
        print(f"   总共找到 {len(selected_images)} 张图片\n")
    
    # 方式3: 通过序列号筛选
    elif sequences:
        print(f"📌 通过序列号筛选: {sequences}")
        for seq in sequences:
            matched = [img for img in images if img.get('sequence') == str(seq) or 
                      img['file_name'].startswith(f"{seq}/")]
            selected_images.extend(matched)
            print(f"   序列 {seq} 找到 {len(matched)} 张图片")
        print(f"   总共找到 {len(selected_images)} 张图片\n")
    
    # 方式4: 通过正则表达式筛选
    elif pattern:
        print(f"📌 通过正则表达式筛选: {pattern}")
        regex = re.compile(pattern)
        selected_images = [img for img in images if regex.search(img['file_name'])]
        print(f"   找到 {len(selected_images)} 张图片\n")
    
    else:
        print("❌ 请指定至少一种筛选方式（image_ids, filenames, sequences, pattern）")
        return None
    
    if not selected_images:
        print("❌ 没有找到匹配的图片")
        return None
    
    # 去重
    selected_images = list({img['id']: img for img in selected_images}.values())
    
    print(f"🖼️  开始可视化 {len(selected_images)} 张图片...\n")
    
    # 统计信息
    stats = {
        'total_visualized': 0,
        'with_annotations': 0,
        'without_annotations': 0,
        'image_not_found': 0,
        'image_read_error': 0
    }
    
    # 可视化每张图片
    for idx, img_info in enumerate(selected_images, 1):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 构建图像路径
        img_path = Path(coco_root) / img_filename
        
        if not img_path.exists():
            print(f"   ⚠️  [{idx}/{len(selected_images)}] 图片不存在: {img_path}")
            stats['image_not_found'] += 1
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ⚠️  [{idx}/{len(selected_images)}] 无法读取图片: {img_path}")
            stats['image_read_error'] += 1
            continue
        
        # 获取标注
        anns = img_id_to_anns.get(img_id, [])
        
        if anns:
            stats['with_annotations'] += 1
        else:
            stats['without_annotations'] += 1
        
        # 绘制边界框
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 转换为整数坐标
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 检查是否超出边界
            is_out_of_bounds = (x < 0 or y < 0 or x + w > img_width or y + h > img_height)
            color = (0, 0, 255) if is_out_of_bounds else (0, 255, 0)
            
            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加bbox信息
            label = f"W:{w:.0f} H:{h:.0f}"
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制中心点
            center_x, center_y = int(x + w/2), int(y + h/2)
            cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), -1)
        
        if show_info:
            # 添加图片信息
            info_lines = [
                f"ID:{img_id} | {Path(img_filename).name}",
                f"Size: {img_width}x{img_height} | Objects: {len(anns)}",
            ]
            
            # 添加时间戳信息（如果有）
            if 'relative_timestamp' in img_info:
                rel_time = img_info['relative_timestamp']
                info_lines.append(f"Time: {rel_time:.3f}s")
            
            # 添加序列信息（如果有）
            if 'sequence' in img_info:
                info_lines.append(f"Seq: {img_info['sequence']}")
            
            # 绘制信息文本
            y_offset = 30
            for line in info_lines:
                cv2.putText(img, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25
        
        # 保存可视化结果
        output_filename = f"{idx:04d}_id{img_id}_{Path(img_filename).name}"
        output_path = output_dir_path / output_filename
        cv2.imwrite(str(output_path), img)
        
        stats['total_visualized'] += 1
        
        # 打印进度
        status_icon = '✅' if anns else '⚪'
        print(f"   {status_icon} [{idx}/{len(selected_images)}] ID:{img_id} | {Path(img_filename).name}")
        if show_info:
            print(f"       标注数: {len(anns)}, 保存至: {output_filename}")
            if 'relative_timestamp' in img_info:
                print(f"       时间戳: {img_info['relative_timestamp']:.6f}s")
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"📊 可视化统计:")
    print(f"{'='*80}")
    print(f"总共可视化: {stats['total_visualized']} 张")
    print(f"有标注: {stats['with_annotations']} 张")
    print(f"无标注: {stats['without_annotations']} 张")
    print(f"图片不存在: {stats['image_not_found']} 张")
    print(f"读取错误: {stats['image_read_error']} 张")
    print(f"{'='*80}\n")
    
    print(f"✅ 可视化完成！")
    print(f"   输出目录: {output_dir_path}")
    print(f"{'='*80}\n")
    
    return stats


def list_available_images(output_root='datasets/fred_coco',
                          modality='rgb',
                          split='train',
                          limit=20):
    """
    列出可用的图片信息
    """
    coco_root_path = Path(output_root) / modality
    ann_file = coco_root_path / 'annotations' / f'instances_{split}.json'
    
    if not ann_file.exists():
        print(f"❌ 标注文件不存在: {ann_file}")
        return
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    
    print(f"\n{'='*80}")
    print(f"📋 可用图片列表 - {modality.upper()} {split.upper()}")
    print(f"{'='*80}")
    print(f"总图片数: {len(images)}")
    print(f"显示前 {min(limit, len(images))} 张:\n")
    
    for i, img in enumerate(images[:limit], 1):
        print(f"{i}. ID:{img['id']:5d} | {img['file_name']}")
        if 'sequence' in img:
            print(f"   序列: {img['sequence']}, 尺寸: {img['width']}x{img['height']}")
        if 'relative_timestamp' in img:
            print(f"   时间戳: {img['relative_timestamp']:.6f}s")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化特定图片的标注')
    parser.add_argument('--coco_root', type=str, default='/mnt/data/datasets/fred',
                        help='FRED原始数据集根目录')
    parser.add_argument('--output_root', type=str, default='datasets/fred_coco',
                        help='COCO数据集根目录')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='模态')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='数据集划分')
    parser.add_argument('--output_dir', type=str, default='specific_visualization',
                        help='输出目录')
    
    # 筛选方式
    parser.add_argument('--image_ids', type=int, nargs='+',
                        help='图片ID列表，如: --image_ids 1 2 3')
    parser.add_argument('--filenames', type=str, nargs='+',
                        help='文件名列表（支持部分匹配），如: --filenames Video_0_16_03_03 Video_1_16_05_12')
    parser.add_argument('--sequences', type=int, nargs='+',
                        help='序列号列表，如: --sequences 1 3 5')
    parser.add_argument('--pattern', type=str,
                        help='文件名正则表达式，如: --pattern "Video_0_16_03_.*"')
    
    # 其他选项
    parser.add_argument('--list', action='store_true',
                        help='列出可用的图片信息')
    parser.add_argument('--list_limit', type=int, default=20,
                        help='列出图片的数量限制')
    parser.add_argument('--no_info', action='store_true',
                        help='不显示详细信息')
    
    args = parser.parse_args()
    
    # 列出可用图片
    if args.list:
        list_available_images(
            output_root=args.output_root,
            modality=args.modality,
            split=args.split,
            limit=args.list_limit
        )
    else:
        # 可视化特定图片
        visualize_specific_images(
            coco_root=args.coco_root,
            output_root=args.output_root,
            modality=args.modality,
            split=args.split,
            image_ids=args.image_ids,
            filenames=args.filenames,
            sequences=args.sequences,
            pattern=args.pattern,
            output_dir=args.output_dir,
            show_info=not args.no_info
        )
