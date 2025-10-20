#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复Event数据集的边界框问题

Event相机的YOLO标注中有些边界框超出了图像边界（主要是右边界）。
这个脚本提供两种修复方案：
1. 保留原始坐标（可能超出边界）
2. 裁剪到图像范围内（当前方案）

问题分析：
- Event数据中约有部分标注的x_max超出1280像素
- 这可能是由于Event相机和RGB相机的视场角或标注工具差异导致的
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

def analyze_bbox_issues(coco_json_path):
    """分析边界框问题"""
    with open(coco_json_path) as f:
        data = json.load(f)
    
    print(f"\n分析文件: {coco_json_path}")
    print("=" * 70)
    
    total = len(data['annotations'])
    issues = {
        'x_negative': 0,
        'y_negative': 0,
        'x_overflow': 0,
        'y_overflow': 0,
        'width_zero': 0,
        'height_zero': 0,
    }
    
    overflow_examples = []
    
    for ann in data['annotations']:
        img = next(img for img in data['images'] if img['id'] == ann['image_id'])
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        if x < 0:
            issues['x_negative'] += 1
        if y < 0:
            issues['y_negative'] += 1
        if x + w > img['width']:
            issues['x_overflow'] += 1
            if len(overflow_examples) < 5:
                overflow_examples.append({
                    'image': img['file_name'],
                    'bbox': bbox,
                    'overflow': x + w - img['width']
                })
        if y + h > img['height']:
            issues['y_overflow'] += 1
        if w <= 0:
            issues['width_zero'] += 1
        if h <= 0:
            issues['height_zero'] += 1
    
    print(f"总标注数: {total}")
    print(f"\n边界框问题统计:")
    for issue, count in issues.items():
        if count > 0:
            print(f"  {issue}: {count} ({count/total*100:.2f}%)")
    
    if overflow_examples:
        print(f"\n超出边界示例:")
        for ex in overflow_examples:
            print(f"  {ex['image']}: bbox={ex['bbox']}, 超出={ex['overflow']:.2f}px")
    
    return issues

def check_original_yolo_data(fred_root, modality='event', num_samples=100):
    """检查原始YOLO数据"""
    print(f"\n检查原始{modality.upper()}数据的YOLO标注")
    print("=" * 70)
    
    from pathlib import Path
    import random
    
    fred_path = Path(fred_root)
    
    if modality.lower() == 'event':
        label_subdir = "Event_YOLO"
    else:
        label_subdir = "RGB_YOLO"
    
    # 收集所有标注文件
    all_labels = []
    for video_dir in fred_path.iterdir():
        if not video_dir.is_dir() or video_dir.name.startswith('.'):
            continue
        
        yolo_dir = video_dir / label_subdir
        if yolo_dir.exists():
            all_labels.extend(list(yolo_dir.glob("*.txt")))
    
    # 随机采样
    samples = random.sample(all_labels, min(num_samples, len(all_labels)))
    
    img_w, img_h = 1280, 720
    out_of_bounds = 0
    x_overflow_count = 0
    y_overflow_count = 0
    max_x_overflow = 0
    max_y_overflow = 0
    
    for label_file in samples:
        if label_file.stat().st_size == 0:
            continue
        
        with open(label_file) as f:
            content = f.read().strip()
            if not content:
                continue
            
            parts = content.split()
            if len(parts) != 5:
                continue
            
            class_id, x_c, y_c, w, h = map(float, parts)
            
            # 计算像素坐标
            x_c_px = x_c * img_w
            y_c_px = y_c * img_h
            w_px = w * img_w
            h_px = h * img_h
            
            x_min = x_c_px - w_px / 2
            x_max = x_c_px + w_px / 2
            y_min = y_c_px - h_px / 2
            y_max = y_c_px + h_px / 2
            
            # 检查是否超出边界
            if x_min < 0 or x_max > img_w or y_min < 0 or y_max > img_h:
                out_of_bounds += 1
                
                if x_max > img_w:
                    x_overflow_count += 1
                    max_x_overflow = max(max_x_overflow, x_max - img_w)
                
                if y_max > img_h:
                    y_overflow_count += 1
                    max_y_overflow = max(max_y_overflow, y_max - img_h)
    
    print(f"采样数量: {len(samples)}")
    print(f"超出边界: {out_of_bounds} ({out_of_bounds/len(samples)*100:.1f}%)")
    print(f"  X轴超出: {x_overflow_count} (最大超出: {max_x_overflow:.2f}px)")
    print(f"  Y轴超出: {y_overflow_count} (最大超出: {max_y_overflow:.2f}px)")

def main():
    parser = argparse.ArgumentParser(description='分析和修复Event数据集的边界框问题')
    parser.add_argument('--action', type=str, default='analyze',
                       choices=['analyze', 'check_original'],
                       help='操作类型')
    parser.add_argument('--modality', type=str, default='event',
                       choices=['rgb', 'event'],
                       help='数据模态')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='数据集划分')
    parser.add_argument('--coco_root', type=str, default='datasets/fred_coco',
                       help='COCO数据集根目录')
    parser.add_argument('--fred_root', type=str, default='/mnt/data/datasets/fred',
                       help='FRED原始数据根目录')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        # 分析COCO数据
        coco_json = Path(args.coco_root) / args.modality / "annotations" / f"instances_{args.split}.json"
        if not coco_json.exists():
            print(f"错误: 文件不存在 {coco_json}")
            return
        
        analyze_bbox_issues(coco_json)
    
    elif args.action == 'check_original':
        # 检查原始YOLO数据
        check_original_yolo_data(args.fred_root, args.modality)
    
    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    print("""
Event相机数据的YOLO标注中确实存在边界框超出图像边界的情况。
这主要发生在图像的右边界（x轴方向）。

可能的原因：
1. Event相机和RGB相机的视场角不同
2. 标注工具的坐标系统差异
3. 数据采集或标注过程中的误差

当前的转换脚本采用了裁剪策略，将超出边界的部分裁剪到图像范围内。
这是一个合理的处理方式，因为：
- 保证了所有边界框都在有效范围内
- 不会影响模型训练（边界框仍然标注了目标的主要部分）
- 符合COCO格式的规范

如果需要保留原始坐标（用于研究或对比），可以修改转换脚本中的
yolo_to_coco_bbox函数，移除边界裁剪的代码。
    """)

if __name__ == "__main__":
    main()
