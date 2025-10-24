#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证时间戳修复是否正确
检查RGB图片和coordinates.txt的时间对应关系
"""

import os
import re
from pathlib import Path


def extract_absolute_timestamp_from_filename(filename, modality='rgb'):
    """
    从文件名中提取绝对时间戳（秒）
    RGB: Video_0_16_03_03.363444.jpg -> 16*3600 + 03*60 + 03.363444
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
    except Exception as e:
        print(f"警告: 无法从文件名 '{filename}' 提取时间戳: {e}")
        return None
    
    return None


def parse_coordinates_file(coord_file):
    """解析coordinates.txt文件"""
    annotations = {}
    
    with open(coord_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                
                timestamp = float(parts[0].strip())
                coords_str = parts[1].strip()
                
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    annotations[timestamp] = coords
            except Exception as e:
                continue
    
    return annotations


def verify_timestamp_mapping(video_id=3):
    """验证时间戳映射关系"""
    
    print("=" * 80)
    print(f"验证视频{video_id}的时间戳映射关系")
    print("=" * 80)
    
    # 路径
    seq_dir = Path(f'/mnt/data/datasets/fred/{video_id}')
    coord_file = seq_dir / 'coordinates.txt'
    rgb_dir = seq_dir / 'PADDED_RGB'
    
    if not coord_file.exists():
        print(f"❌ coordinates.txt不存在: {coord_file}")
        return
    
    if not rgb_dir.exists():
        print(f"❌ PADDED_RGB目录不存在: {rgb_dir}")
        return
    
    # 读取coordinates.txt
    annotations = parse_coordinates_file(coord_file)
    print(f"\n📋 coordinates.txt:")
    print(f"   总标注数: {len(annotations)}")
    print(f"   时间范围: {min(annotations.keys()):.6f}s ~ {max(annotations.keys()):.6f}s")
    
    # 读取RGB图片
    rgb_images = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])
    print(f"\n🖼️  PADDED_RGB:")
    print(f"   总图片数: {len(rgb_images)}")
    
    if not rgb_images:
        print("❌ 没有找到RGB图片")
        return
    
    # 获取第一张和最后一张图片的时间戳
    first_image = rgb_images[0]
    last_image = rgb_images[-1]
    
    first_abs_time = extract_absolute_timestamp_from_filename(first_image.name)
    last_abs_time = extract_absolute_timestamp_from_filename(last_image.name)
    
    print(f"\n⏱️  RGB图片时间戳:")
    print(f"   第一张: {first_image.name}")
    print(f"           绝对时间: {first_abs_time:.6f}s")
    print(f"   最后一张: {last_image.name}")
    print(f"           绝对时间: {last_abs_time:.6f}s")
    print(f"   时间跨度: {last_abs_time - first_abs_time:.6f}s")
    
    # 关键验证：第一张RGB图片应该对应0时刻
    print(f"\n✅ 关键假设验证:")
    print(f"   假设: RGB的第一张图被认为是0时刻")
    print(f"   即: coordinates.txt中的时间戳是相对于第一张RGB图片的相对时间")
    
    # 计算video_start_time（修复后的逻辑）
    video_start_time = first_abs_time
    print(f"\n🎬 视频起始时间 (video_start_time):")
    print(f"   {video_start_time:.6f}s (第一张RGB图片的绝对时间)")
    
    # 验证几个样本
    print(f"\n🔍 样本验证 (前5张图片):")
    print(f"{'序号':<4} {'图片文件名':<35} {'绝对时间':<12} {'相对时间':<12} {'最近标注':<12} {'时间差':<10}")
    print("-" * 100)
    
    for i, img_path in enumerate(rgb_images[:5]):
        abs_time = extract_absolute_timestamp_from_filename(img_path.name)
        if abs_time is None:
            continue
        
        # 计算相对时间（修复后的逻辑）
        relative_time = abs_time - video_start_time
        
        # 找到最接近的标注
        closest_anno_time = min(annotations.keys(), key=lambda t: abs(t - relative_time))
        time_diff = abs(closest_anno_time - relative_time)
        
        print(f"{i+1:<4} {img_path.name:<35} {abs_time:<12.6f} {relative_time:<12.6f} {closest_anno_time:<12.6f} {time_diff:<10.6f}")
    
    # 统计匹配情况
    print(f"\n📊 匹配统计 (容差=0.05s):")
    tolerance = 0.05
    matched = 0
    unmatched = 0
    
    for img_path in rgb_images:
        abs_time = extract_absolute_timestamp_from_filename(img_path.name)
        if abs_time is None:
            unmatched += 1
            continue
        
        relative_time = abs_time - video_start_time
        
        # 找到最接近的标注
        closest_anno_time = min(annotations.keys(), key=lambda t: abs(t - relative_time))
        time_diff = abs(closest_anno_time - relative_time)
        
        if time_diff <= tolerance:
            matched += 1
        else:
            unmatched += 1
    
    total = matched + unmatched
    print(f"   总图片数: {total}")
    print(f"   匹配数: {matched} ({matched/total*100:.2f}%)")
    print(f"   未匹配数: {unmatched} ({unmatched/total*100:.2f}%)")
    
    # 检查是否有改善
    print(f"\n💡 结论:")
    if matched / total > 0.9:
        print(f"   ✅ 时间戳映射正确！匹配率 {matched/total*100:.2f}%")
    elif matched / total > 0.5:
        print(f"   ⚠️  时间戳映射部分正确，匹配率 {matched/total*100:.2f}%")
        print(f"   建议检查未匹配的样本")
    else:
        print(f"   ❌ 时间戳映射可能仍有问题，匹配率仅 {matched/total*100:.2f}%")
        print(f"   需要进一步调查")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='验证时间戳修复')
    parser.add_argument('--video_id', type=int, default=3,
                       help='视频序列ID')
    
    args = parser.parse_args()
    
    verify_timestamp_mapping(args.video_id)
