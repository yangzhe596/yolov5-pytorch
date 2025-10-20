#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比RGB和Event数据的边界框
检查转换是否正确
"""

import json
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def check_bbox_conversion(modality='rgb', num_samples=5):
    """检查边界框转换是否正确"""
    
    print("=" * 80)
    print(f"检查{modality.upper()}数据的边界框转换")
    print("=" * 80)
    
    # 加载COCO标注
    coco_file = f'datasets/fred_coco/{modality}/annotations/instances_train.json'
    with open(coco_file) as f:
        coco_data = json.load(f)
    
    # 检查样本
    errors = []
    
    for i in range(min(num_samples, len(coco_data['images']))):
        img_info = coco_data['images'][i]
        ann = coco_data['annotations'][i]
        
        # 从文件名提取原始路径
        filename = img_info['file_name']
        parts = filename.split('_')
        video_id = parts[0]
        
        # 构建原始YOLO文件路径
        if modality == 'rgb':
            original_name = '_'.join(parts[1:]).replace('.jpg', '.txt')
            yolo_file = f'/mnt/data/datasets/fred/{video_id}/RGB_YOLO/{original_name}'
        else:
            original_name = '_'.join(parts[1:]).replace('.png', '.txt')
            yolo_file = f'/mnt/data/datasets/fred/{video_id}/Event_YOLO/{original_name}'
        
        if not os.path.exists(yolo_file):
            print(f"⚠️  YOLO文件不存在: {yolo_file}")
            continue
        
        # 读取YOLO标注
        with open(yolo_file) as f:
            yolo_content = f.read().strip()
            if not yolo_content:
                continue
            
            yolo_parts = yolo_content.split()
            if len(yolo_parts) != 5:
                continue
            
            class_id, x_c, y_c, w, h = map(float, yolo_parts)
        
        # 计算期望的COCO bbox
        img_w, img_h = img_info['width'], img_info['height']
        x_c_px = x_c * img_w
        y_c_px = y_c * img_h
        w_px = w * img_w
        h_px = h * img_h
        
        x_expected = x_c_px - w_px / 2
        y_expected = y_c_px - h_px / 2
        
        # 检查是否需要裁剪
        x_clipped = max(0, x_expected)
        y_clipped = max(0, y_expected)
        w_clipped = min(w_px, img_w - x_clipped)
        h_clipped = min(h_px, img_h - y_clipped)
        
        # 对比实际COCO bbox
        coco_bbox = ann['bbox']
        
        # 判断是否匹配
        tolerance = 0.1
        match = (abs(x_clipped - coco_bbox[0]) < tolerance and
                abs(y_clipped - coco_bbox[1]) < tolerance and
                abs(w_clipped - coco_bbox[2]) < tolerance and
                abs(h_clipped - coco_bbox[3]) < tolerance)
        
        was_clipped = (abs(x_expected - x_clipped) > 0.01 or
                      abs(y_expected - y_clipped) > 0.01 or
                      abs(w_px - w_clipped) > 0.01 or
                      abs(h_px - h_clipped) > 0.01)
        
        status = '✅' if match else '❌'
        clip_mark = '✂️' if was_clipped else '  '
        
        print(f"{status} {clip_mark} 样本 {i+1}: {filename[:50]}")
        
        if not match:
            errors.append({
                'filename': filename,
                'yolo': yolo_content,
                'expected': [x_clipped, y_clipped, w_clipped, h_clipped],
                'actual': coco_bbox
            })
            print(f"     原始YOLO: {yolo_content}")
            print(f"     期望COCO: [{x_clipped:.2f}, {y_clipped:.2f}, {w_clipped:.2f}, {h_clipped:.2f}]")
            print(f"     实际COCO: {coco_bbox}")
        
        if was_clipped:
            print(f"     (边界框被裁剪: 原始超出边界)")
    
    print(f"\n检查结果: {num_samples - len(errors)}/{num_samples} 正确")
    
    if errors:
        print(f"\n发现 {len(errors)} 个不匹配的样本！")
    else:
        print(f"\n✅ 所有样本转换正确！")
    
    return errors

def visualize_comparison(modality='rgb'):
    """可视化对比原始YOLO和转换后的COCO"""
    
    print(f"\n生成{modality.upper()}数据的可视化对比...")
    
    # 加载COCO数据
    coco_file = f'datasets/fred_coco/{modality}/annotations/instances_train.json'
    with open(coco_file) as f:
        coco_data = json.load(f)
    
    # 选择一个样本
    img_info = coco_data['images'][0]
    ann = coco_data['annotations'][0]
    
    # 加载图片
    img_path = f'datasets/fred_coco/{modality}/train/{img_info["file_name"]}'
    img = Image.open(img_path)
    
    # 从文件名提取原始路径
    filename = img_info['file_name']
    parts = filename.split('_')
    video_id = parts[0]
    
    if modality == 'rgb':
        original_name = '_'.join(parts[1:]).replace('.jpg', '.txt')
        yolo_file = f'/mnt/data/datasets/fred/{video_id}/RGB_YOLO/{original_name}'
    else:
        original_name = '_'.join(parts[1:]).replace('.png', '.txt')
        yolo_file = f'/mnt/data/datasets/fred/{video_id}/Event_YOLO/{original_name}'
    
    # 读取YOLO标注
    with open(yolo_file) as f:
        yolo_content = f.read().strip()
        yolo_parts = yolo_content.split()
        class_id, x_c, y_c, w, h = map(float, yolo_parts)
    
    # 计算YOLO bbox（未裁剪）
    img_w, img_h = img_info['width'], img_info['height']
    x_c_px = x_c * img_w
    y_c_px = y_c * img_h
    w_px = w * img_w
    h_px = h * img_h
    x_yolo = x_c_px - w_px / 2
    y_yolo = y_c_px - h_px / 2
    
    # COCO bbox（可能已裁剪）
    coco_bbox = ann['bbox']
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：原始YOLO bbox
    axes[0].imshow(img)
    axes[0].set_title(f'原始YOLO标注\n{filename}', fontsize=12)
    axes[0].axis('off')
    
    rect_yolo = patches.Rectangle(
        (x_yolo, y_yolo), w_px, h_px,
        linewidth=3, edgecolor='blue', facecolor='none',
        label='YOLO bbox'
    )
    axes[0].add_patch(rect_yolo)
    axes[0].legend()
    
    # 右图：COCO bbox
    axes[1].imshow(img)
    axes[1].set_title(f'转换后COCO标注\n{filename}', fontsize=12)
    axes[1].axis('off')
    
    rect_coco = patches.Rectangle(
        (coco_bbox[0], coco_bbox[1]), coco_bbox[2], coco_bbox[3],
        linewidth=3, edgecolor='red', facecolor='none',
        label='COCO bbox'
    )
    axes[1].add_patch(rect_coco)
    axes[1].legend()
    
    plt.tight_layout()
    
    # 保存
    output_path = f'bbox_comparison_{modality}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_path}")
    
    # 打印信息
    print(f"\n详细信息:")
    print(f"  原始YOLO: {yolo_content}")
    print(f"  YOLO bbox (像素): x={x_yolo:.2f}, y={y_yolo:.2f}, w={w_px:.2f}, h={h_px:.2f}")
    print(f"  COCO bbox: {coco_bbox}")
    print(f"  是否被裁剪: {abs(w_px - coco_bbox[2]) > 0.1 or abs(h_px - coco_bbox[3]) > 0.1}")
    
    try:
        plt.show()
    except:
        print("  (无法显示图片，已保存到文件)")
    
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对比RGB和Event数据的边界框')
    parser.add_argument('--modality', type=str, default='rgb',
                       choices=['rgb', 'event', 'both'],
                       help='选择模态')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='检查样本数量')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化对比')
    
    args = parser.parse_args()
    
    modalities = ['rgb', 'event'] if args.modality == 'both' else [args.modality]
    
    for mod in modalities:
        errors = check_bbox_conversion(mod, args.num_samples)
        
        if args.visualize:
            visualize_comparison(mod)
        
        print()
