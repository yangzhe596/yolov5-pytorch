#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化RGB数据的两个标注来源
对比RGB_YOLO和coordinates.txt
"""

import os
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_closest_coordinate(timestamp_str, coord_file):
    """在coordinates.txt中查找最接近的时间戳"""
    
    # 从文件名提取时间戳
    # 格式: Video_3_16_46_03.278530.jpg
    # 提取: 16:46:03.278530
    match = re.search(r'(\d+)_(\d+)_(\d+)\.(\d+)', timestamp_str)
    if not match:
        return None
    
    h, m, s, ms = match.groups()
    
    # 尝试不同的时间基准
    # 假设1: 相对于视频开始的秒数（从16:45:00开始）
    relative_time_1 = int(m) * 60 + float(f"{s}.{ms}")
    
    # 假设2: 仅使用秒和毫秒部分
    relative_time_2 = float(f"{s}.{ms}")
    
    # 读取coordinates.txt
    with open(coord_file) as f:
        lines = f.readlines()
    
    # 查找最接近的时间戳
    best_match = None
    min_diff = float('inf')
    
    for line in lines:
        if ':' in line:
            parts = line.strip().split(':')
            ts = float(parts[0])
            coords_str = parts[1].strip()
            
            # 尝试两种时间基准
            diff1 = abs(ts - relative_time_1)
            diff2 = abs(ts - relative_time_2)
            
            diff = min(diff1, diff2)
            
            if diff < min_diff:
                min_diff = diff
                coord_values = [float(x.strip()) for x in coords_str.split(',')]
                if len(coord_values) == 4:
                    best_match = {
                        'timestamp': ts,
                        'bbox': coord_values,
                        'diff': diff,
                        'line': line.strip()
                    }
    
    return best_match

def visualize_both_annotations(image_path, yolo_path, coord_file, video_id):
    """可视化两个标注源"""
    
    print("=" * 80)
    print(f"可视化对比: {os.path.basename(image_path)}")
    print("=" * 80)
    
    # 加载图片
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    print(f"\n图片信息:")
    print(f"  路径: {image_path}")
    print(f"  尺寸: {img_w} x {img_h}")
    
    # 读取RGB_YOLO标注
    yolo_bbox = None
    if os.path.exists(yolo_path) and os.path.getsize(yolo_path) > 0:
        with open(yolo_path) as f:
            yolo_content = f.read().strip()
            if yolo_content:
                parts = yolo_content.split()
                class_id, x_c, y_c, w, h = map(float, parts)
                
                # 转换为像素坐标
                x_c_px = x_c * img_w
                y_c_px = y_c * img_h
                w_px = w * img_w
                h_px = h * img_h
                
                xmin = x_c_px - w_px / 2
                ymin = y_c_px - h_px / 2
                xmax = x_c_px + w_px / 2
                ymax = y_c_px + h_px / 2
                
                yolo_bbox = [xmin, ymin, xmax, ymax]
                
                print(f"\nRGB_YOLO标注:")
                print(f"  原始: {yolo_content}")
                print(f"  边界框: xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
                print(f"  尺寸: {w_px:.1f} × {h_px:.1f}")
    
    # 查找coordinates.txt中的对应标注
    coord_match = find_closest_coordinate(os.path.basename(image_path), coord_file)
    coord_bbox = None
    
    if coord_match:
        coord_bbox = coord_match['bbox']
        xmin, ymin, xmax, ymax = coord_bbox
        
        print(f"\ncoordinates.txt标注:")
        print(f"  原始: {coord_match['line']}")
        print(f"  时间差: {coord_match['diff']:.6f} 秒")
        print(f"  边界框: xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
        print(f"  尺寸: {xmax-xmin:.1f} × {ymax-ymin:.1f}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原图
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # RGB_YOLO标注
    axes[1].imshow(img)
    axes[1].set_title('RGB_YOLO Annotation', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    if yolo_bbox:
        xmin, ymin, xmax, ymax = yolo_bbox
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=3, edgecolor='red', facecolor='none',
            label='RGB_YOLO'
        )
        axes[1].add_patch(rect)
        axes[1].text(xmin, ymin - 10, 'RGB_YOLO', 
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8))
        axes[1].legend(loc='upper right')
    
    # coordinates.txt标注
    axes[2].imshow(img)
    axes[2].set_title('coordinates.txt Annotation', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    if coord_bbox:
        xmin, ymin, xmax, ymax = coord_bbox
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=3, edgecolor='blue', facecolor='none',
            label='coordinates.txt'
        )
        axes[2].add_patch(rect)
        axes[2].text(xmin, ymin - 10, 'coordinates.txt',
                    color='blue', fontsize=12, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8))
        axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存
    output_path = f'annotation_comparison_{video_id}_{os.path.basename(image_path).replace(".jpg", ".png")}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("  (无法显示图片)")
    
    plt.close()
    
    # 分析
    print("\n" + "=" * 80)
    print("分析:")
    print("=" * 80)
    
    if yolo_bbox and coord_bbox:
        yolo_center_x = (yolo_bbox[0] + yolo_bbox[2]) / 2
        yolo_center_y = (yolo_bbox[1] + yolo_bbox[3]) / 2
        coord_center_x = (coord_bbox[0] + coord_bbox[2]) / 2
        coord_center_y = (coord_bbox[1] + coord_bbox[3]) / 2
        
        distance = ((yolo_center_x - coord_center_x)**2 + 
                   (yolo_center_y - coord_center_y)**2)**0.5
        
        print(f"两个标注的中心点距离: {distance:.1f} 像素")
        
        if distance > 100:
            print("⚠️  两个标注位置相差很大，可能标注的是不同的目标！")
        elif distance > 50:
            print("⚠️  两个标注位置有明显差异")
        else:
            print("✓  两个标注位置接近，可能是同一目标")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化RGB数据的两个标注源')
    parser.add_argument('--video_id', type=int, default=3,
                       help='视频序列ID')
    parser.add_argument('--image_name', type=str, 
                       default='Video_3_16_46_03.278530.jpg',
                       help='图片文件名')
    
    args = parser.parse_args()
    
    # 构建路径
    fred_root = '/mnt/data/datasets/fred'
    image_path = f'{fred_root}/{args.video_id}/RGB/{args.image_name}'
    yolo_path = f'{fred_root}/{args.video_id}/RGB_YOLO/{args.image_name.replace(".jpg", ".txt")}'
    coord_file = f'{fred_root}/{args.video_id}/coordinates.txt'
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 {image_path}")
        return
    
    if not os.path.exists(yolo_path):
        print(f"错误: YOLO标注不存在 {yolo_path}")
        return
    
    if not os.path.exists(coord_file):
        print(f"错误: coordinates.txt不存在 {coord_file}")
        return
    
    # 可视化
    visualize_both_annotations(image_path, yolo_path, coord_file, args.video_id)
    
    print("\n" + "=" * 80)
    print("请查看生成的可视化图片，确认哪个标注源是正确的。")
    print("=" * 80)

if __name__ == "__main__":
    main()
