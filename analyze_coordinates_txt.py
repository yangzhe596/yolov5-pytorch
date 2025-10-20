#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析coordinates.txt文件
理解它与YOLO标注的关系
"""

import os
import re
from pathlib import Path

def analyze_coordinates_file(video_id=3):
    """分析coordinates.txt文件"""
    
    print("=" * 80)
    print(f"分析视频{video_id}的coordinates.txt文件")
    print("=" * 80)
    
    coord_file = f'/mnt/data/datasets/fred/{video_id}/coordinates.txt'
    
    if not os.path.exists(coord_file):
        print(f"文件不存在: {coord_file}")
        return
    
    # 读取coordinates.txt
    with open(coord_file) as f:
        coord_lines = f.readlines()
    
    print(f"\ncoordinates.txt统计:")
    print(f"  总行数: {len(coord_lines)}")
    
    # 解析前几行
    print(f"\n前10行内容:")
    for i, line in enumerate(coord_lines[:10]):
        print(f"  {i+1}. {line.strip()}")
    
    # 解析格式
    print(f"\n格式分析:")
    if coord_lines:
        first_line = coord_lines[0].strip()
        parts = first_line.split(':')
        timestamp = parts[0]
        coords = parts[1].strip()
        
        print(f"  时间戳: {timestamp} (相对时间，单位：秒)")
        print(f"  坐标: {coords} (格式: xmin, ymin, xmax, ymax)")
        
        # 解析坐标
        coord_values = [float(x.strip()) for x in coords.split(',')]
        xmin, ymin, xmax, ymax = coord_values
        
        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        
        print(f"\n  边界框信息:")
        print(f"    左上角: ({xmin}, {ymin})")
        print(f"    右下角: ({xmax}, {ymax})")
        print(f"    中心点: ({x_center:.1f}, {y_center:.1f})")
        print(f"    宽×高: {width:.1f} × {height:.1f}")
    
    # 统计RGB_YOLO文件数量
    rgb_yolo_dir = f'/mnt/data/datasets/fred/{video_id}/RGB_YOLO'
    rgb_yolo_files = [f for f in os.listdir(rgb_yolo_dir) 
                     if f.endswith('.txt') and os.path.getsize(os.path.join(rgb_yolo_dir, f)) > 0]
    
    print(f"\nRGB_YOLO标注文件:")
    print(f"  有效标注数: {len(rgb_yolo_files)}")
    
    print(f"\n对比:")
    print(f"  coordinates.txt: {len(coord_lines)} 条")
    print(f"  RGB_YOLO: {len(rgb_yolo_files)} 个有效标注")
    print(f"  差异: {abs(len(coord_lines) - len(rgb_yolo_files))} 条")
    
    # 结论
    print(f"\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("""
coordinates.txt 是原始的边界框坐标文件，包含：
  - 时间戳（相对时间，秒）
  - 边界框坐标（xmin, ymin, xmax, ymax，像素格式）

RGB_YOLO/ 目录包含：
  - 每帧图片对应的YOLO格式标注
  - 格式：class_id x_center y_center width height（归一化）

两者的关系：
  - coordinates.txt 可能是原始标注或跟踪结果
  - RGB_YOLO 是从coordinates.txt转换而来的YOLO格式
  - 数量可能不完全一致（某些帧可能没有目标）

当前的COCO转换使用的是 RGB_YOLO 目录中的标注，这是正确的。
    """)

def compare_with_yolo(video_id=3, sample_timestamp=None):
    """对比coordinates.txt和YOLO标注"""
    
    print("\n" + "=" * 80)
    print(f"对比coordinates.txt和YOLO标注")
    print("=" * 80)
    
    coord_file = f'/mnt/data/datasets/fred/{video_id}/coordinates.txt'
    
    # 读取coordinates.txt
    with open(coord_file) as f:
        coord_data = {}
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                ts = float(parts[0])
                coords_str = parts[1].strip()
                coord_values = [float(x.strip()) for x in coords_str.split(',')]
                if len(coord_values) == 4:
                    coord_data[ts] = coord_values
    
    print(f"\ncoordinates.txt: {len(coord_data)} 条记录")
    
    # 读取RGB_YOLO标注
    rgb_yolo_dir = f'/mnt/data/datasets/fred/{video_id}/RGB_YOLO'
    yolo_files = {}
    
    for filename in os.listdir(rgb_yolo_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(rgb_yolo_dir, filename)
            if os.path.getsize(filepath) > 0:
                with open(filepath) as f:
                    content = f.read().strip()
                    if content:
                        yolo_files[filename] = content
    
    print(f"RGB_YOLO: {len(yolo_files)} 个有效标注")
    
    # 随机选择几个YOLO文件，看看能否在coordinates.txt中找到对应的
    import random
    random.seed(42)
    
    sample_files = random.sample(list(yolo_files.keys()), min(5, len(yolo_files)))
    
    print(f"\n随机抽样对比:")
    for filename in sample_files:
        # 从文件名提取时间戳
        # 格式: Video_3_16_46_03.278530.txt
        match = re.search(r'(\d+)_(\d+)_(\d+)\.(\d+)', filename)
        if match:
            h, m, s, ms = match.groups()
            # 这是绝对时间，无法直接对应coordinates.txt中的相对时间
            print(f"\n  文件: {filename}")
            print(f"    时间: {h}:{m}:{s}.{ms}")
            print(f"    YOLO: {yolo_files[filename]}")
            print(f"    (无法直接对应coordinates.txt中的相对时间戳)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析coordinates.txt文件')
    parser.add_argument('--video_id', type=int, default=3,
                       help='视频序列ID')
    
    args = parser.parse_args()
    
    analyze_coordinates_file(args.video_id)
    compare_with_yolo(args.video_id)
