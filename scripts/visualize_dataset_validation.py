#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的数据集可视化验证脚本
导出可视化图片，验证数据集标注的正确性
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from datetime import datetime


def visualize_and_validate(coco_root='/mnt/data/datasets/fred', 
                           output_root='datasets/fred_coco',
                           modality='rgb', 
                           split='train', 
                           num_samples=20,
                           output_dir='dataset_validation',
                           seed=42):
    """
    可视化并验证COCO数据集
    
    Args:
        coco_root: FRED原始数据集根目录
        output_root: COCO数据集根目录
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
        num_samples: 可视化的样本数量
        output_dir: 输出目录
        seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
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
    print(f"📊 数据集可视化验证 - {modality.upper()} {split.upper()}")
    print(f"{'='*80}")
    print(f"标注文件: {ann_file}")
    print(f"输出目录: {output_dir_path}")
    print(f"样本数量: {num_samples}")
    print(f"{'='*80}\n")
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    print(f"📈 数据集统计:")
    print(f"   图像数量: {len(images)}")
    print(f"   标注数量: {len(annotations)}")
    print(f"   类别数量: {len(categories)}")
    for cat in categories:
        print(f"      - {cat['name']} (ID: {cat['id']})")
    print()
    
    # 创建image_id到annotations的映射
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # 统计信息
    stats = {
        'total_images': len(images),
        'total_annotations': len(annotations),
        'images_with_annotations': len(img_id_to_anns),
        'images_without_annotations': len(images) - len(img_id_to_anns),
        'valid_bboxes': 0,
        'out_of_bounds': 0,
        'too_small': 0,
        'too_large': 0,
        'bbox_sizes': [],
        'bbox_areas': []
    }
    
    # 随机选择样本
    if len(images) > num_samples:
        sample_images = random.sample(images, num_samples)
    else:
        sample_images = images
    
    print(f"🖼️  开始可视化 {len(sample_images)} 个样本...\n")
    
    # 创建验证报告
    validation_report = []
    
    # 可视化每个样本
    for idx, img_info in enumerate(sample_images, 1):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 构建图像路径（相对于FRED根目录）
        img_path = Path(coco_root) / img_filename
        
        if not img_path.exists():
            print(f"   ⚠️  [{idx}/{len(sample_images)}] 图片不存在: {img_path}")
            validation_report.append({
                'image_id': img_id,
                'filename': img_filename,
                'status': 'ERROR',
                'message': 'Image file not found'
            })
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ⚠️  [{idx}/{len(sample_images)}] 无法读取图片: {img_path}")
            validation_report.append({
                'image_id': img_id,
                'filename': img_filename,
                'status': 'ERROR',
                'message': 'Cannot read image'
            })
            continue
        
        # 验证图像尺寸
        actual_height, actual_width = img.shape[:2]
        if actual_width != img_width or actual_height != img_height:
            print(f"   ⚠️  [{idx}/{len(sample_images)}] 图像尺寸不匹配!")
            print(f"       标注: {img_width}x{img_height}, 实际: {actual_width}x{actual_height}")
        
        # 获取标注
        anns = img_id_to_anns.get(img_id, [])
        
        # 验证信息
        bbox_issues = []
        
        # 绘制边界框
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 转换为整数坐标
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 验证bbox
            is_valid = True
            issues = []
            
            # 检查是否超出边界
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                is_valid = False
                issues.append('out_of_bounds')
                stats['out_of_bounds'] += 1
            
            # 检查是否太小
            if w < 5 or h < 5:
                issues.append('too_small')
                stats['too_small'] += 1
            
            # 检查是否太大
            if w > img_width * 0.9 or h > img_height * 0.9:
                issues.append('too_large')
                stats['too_large'] += 1
            
            if is_valid:
                stats['valid_bboxes'] += 1
            
            # 记录bbox尺寸
            stats['bbox_sizes'].append((w, h))
            stats['bbox_areas'].append(w * h)
            
            if issues:
                bbox_issues.append({
                    'bbox': bbox,
                    'issues': issues
                })
            
            # 绘制矩形（有问题的用红色，正常的用绿色）
            color = (0, 0, 255) if issues else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加bbox信息
            label = f"W:{w:.0f} H:{h:.0f}"
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制中心点
            center_x, center_y = int(x + w/2), int(y + h/2)
            cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # 添加图片信息
        info_lines = [
            f"ID:{img_id} | {Path(img_filename).name}",
            f"Size: {img_width}x{img_height} | Objects: {len(anns)}",
        ]
        
        # 添加时间戳信息（如果有）
        if 'relative_timestamp' in img_info:
            rel_time = img_info['relative_timestamp']
            info_lines.append(f"Time: {rel_time:.3f}s")
        
        # 绘制信息文本
        y_offset = 30
        for line in info_lines:
            cv2.putText(img, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        # 如果有问题，添加警告
        if bbox_issues:
            warning_text = f"WARNING: {len(bbox_issues)} bbox issues!"
            cv2.putText(img, warning_text, (10, img_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存可视化结果
        output_filename = f"{idx:04d}_{Path(img_filename).stem}.jpg"
        output_path = output_dir_path / output_filename
        cv2.imwrite(str(output_path), img)
        
        # 记录验证结果
        status = 'WARNING' if bbox_issues else 'OK'
        validation_report.append({
            'image_id': img_id,
            'filename': img_filename,
            'num_annotations': len(anns),
            'status': status,
            'bbox_issues': bbox_issues if bbox_issues else None
        })
        
        # 打印进度
        status_icon = '⚠️ ' if bbox_issues else '✅'
        print(f"   {status_icon} [{idx}/{len(sample_images)}] {Path(img_filename).name}")
        print(f"       标注数: {len(anns)}, 保存至: {output_filename}")
        if bbox_issues:
            for issue in bbox_issues:
                print(f"       问题: {', '.join(issue['issues'])} - bbox: {issue['bbox']}")
    
    # 计算统计信息
    if stats['bbox_sizes']:
        bbox_widths = [s[0] for s in stats['bbox_sizes']]
        bbox_heights = [s[1] for s in stats['bbox_sizes']]
        
        print(f"\n{'='*80}")
        print(f"📊 边界框统计:")
        print(f"{'='*80}")
        print(f"总标注数: {stats['total_annotations']}")
        print(f"有效边界框: {stats['valid_bboxes']}")
        print(f"超出边界: {stats['out_of_bounds']}")
        print(f"过小 (<5px): {stats['too_small']}")
        print(f"过大 (>90%): {stats['too_large']}")
        print(f"\n边界框尺寸统计:")
        print(f"  宽度: 平均={np.mean(bbox_widths):.1f}, "
              f"中位数={np.median(bbox_widths):.1f}, "
              f"最小={np.min(bbox_widths):.1f}, "
              f"最大={np.max(bbox_widths):.1f}")
        print(f"  高度: 平均={np.mean(bbox_heights):.1f}, "
              f"中位数={np.median(bbox_heights):.1f}, "
              f"最小={np.min(bbox_heights):.1f}, "
              f"最大={np.max(bbox_heights):.1f}")
        print(f"  面积: 平均={np.mean(stats['bbox_areas']):.1f}, "
              f"中位数={np.median(stats['bbox_areas']):.1f}")
        print(f"{'='*80}\n")
    
    # 保存验证报告
    report_file = output_dir_path / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'modality': modality,
            'split': split,
            'num_samples': len(sample_images),
            'statistics': stats,
            'validation_results': validation_report
        }, f, indent=2)
    
    print(f"📄 验证报告已保存: {report_file}")
    
    # 生成HTML报告
    generate_html_report(output_dir_path, modality, split, validation_report, stats)
    
    print(f"\n✅ 可视化验证完成！")
    print(f"   输出目录: {output_dir_path}")
    print(f"   可视化图片: {len(sample_images)} 张")
    print(f"   验证报告: validation_report.json")
    print(f"   HTML报告: validation_report.html")
    print(f"{'='*80}\n")
    
    return stats, validation_report


def generate_html_report(output_dir, modality, split, validation_report, stats):
    """生成HTML可视化报告"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数据集验证报告 - {modality.upper()} {split.upper()}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stat-item {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background-color: #ecf0f1;
            border-radius: 3px;
        }}
        .stat-label {{
            font-weight: bold;
            color: #34495e;
        }}
        .stat-value {{
            color: #2980b9;
            font-size: 1.2em;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        .image-card {{
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        .image-info {{
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .status-ok {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #e67e22;
            font-weight: bold;
        }}
        .status-error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .issue {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 5px 10px;
            margin-top: 5px;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 数据集验证报告</h1>
        <p>模态: {modality.upper()} | 数据集: {split.upper()} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h2>统计信息</h2>
        <div class="stat-item">
            <span class="stat-label">总图像数:</span>
            <span class="stat-value">{stats['total_images']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">总标注数:</span>
            <span class="stat-value">{stats['total_annotations']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">有效边界框:</span>
            <span class="stat-value">{stats['valid_bboxes']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">超出边界:</span>
            <span class="stat-value">{stats['out_of_bounds']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">过小边界框:</span>
            <span class="stat-value">{stats['too_small']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">过大边界框:</span>
            <span class="stat-value">{stats['too_large']}</span>
        </div>
    </div>
    
    <div class="gallery">
"""
    
    # 添加图片卡片
    for idx, result in enumerate(validation_report, 1):
        if result['status'] == 'ERROR':
            continue
        
        img_filename = f"{idx:04d}_{Path(result['filename']).stem}.jpg"
        status_class = f"status-{result['status'].lower()}"
        
        html_content += f"""
        <div class="image-card">
            <img src="{img_filename}" alt="{result['filename']}">
            <div class="image-info">
                <p><strong>文件:</strong> {Path(result['filename']).name}</p>
                <p><strong>图像ID:</strong> {result['image_id']}</p>
                <p><strong>标注数:</strong> {result['num_annotations']}</p>
                <p><strong>状态:</strong> <span class="{status_class}">{result['status']}</span></p>
"""
        
        if result.get('bbox_issues'):
            html_content += """
                <div class="issue">
                    <strong>⚠️ 边界框问题:</strong><br>
"""
            for issue in result['bbox_issues']:
                html_content += f"                    - {', '.join(issue['issues'])}: {issue['bbox']}<br>\n"
            html_content += """
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # 保存HTML文件
    html_file = output_dir / 'validation_report.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML报告已保存: {html_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据集可视化验证')
    parser.add_argument('--coco_root', type=str, default='/mnt/data/datasets/fred',
                        help='FRED原始数据集根目录')
    parser.add_argument('--output_root', type=str, default='datasets/fred_coco',
                        help='COCO数据集根目录')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event', 'both'],
                        help='模态')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'],
                        help='数据集划分')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='每个划分的可视化样本数量')
    parser.add_argument('--output_dir', type=str, default='dataset_validation',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 处理模态
    modalities = ['rgb', 'event'] if args.modality == 'both' else [args.modality]
    
    # 处理数据集划分
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    # 执行可视化验证
    for modality in modalities:
        for split in splits:
            print(f"\n{'#'*80}")
            print(f"# 处理: {modality.upper()} - {split.upper()}")
            print(f"{'#'*80}\n")
            
            visualize_and_validate(
                coco_root=args.coco_root,
                output_root=args.output_root,
                modality=modality,
                split=split,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                seed=args.seed
            )
    
    print(f"\n{'='*80}")
    print(f"✅ 所有验证完成！")
    print(f"{'='*80}\n")
