#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Fusion 数据集的兼容性
"""

import json
from pathlib import Path
import sys

def test_fusion_compatibility(json_path):
    """测试 Fusion 数据集的兼容性"""
    print("="*70)
    print(f"测试 Fusion 数据集: {json_path}")
    print("="*70)
    
    if not Path(json_path).exists():
        print(f"❌ 文件不存在: {json_path}")
        return False
    
    # 加载 COCO 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images = coco_data.get('images', [])
    
    if not images:
        print("⚠️  警告: 没有图像数据")
        return True
    
    print(f"图像总数: {len(images)}")
    
    # 统计字段
    has_file_name = 0
    has_rgb_file_name = 0
    has_event_file_name = 0
    image_types = {
        'dual': 0,
        'rgb': 0,
        'event': 0
    }
    
    for img in images[:10]:  # 只检查前10张
        if 'file_name' in img:
            has_file_name += 1
        if 'rgb_file_name' in img:
            has_rgb_file_name += 1
        if 'event_file_name' in img:
            has_event_file_name += 1
        
        # 检查 modality 或 status
        modality = img.get('modality') or img.get('status')
        if modality:
            if modality in image_types:
                image_types[modality] += 1
    
    print(f"\n文件名字段:")
    print(f"  file_name: {has_file_name}/10")
    print(f"  rgb_file_name: {has_rgb_file_name}/10")
    print(f"  event_file_name: {has_event_file_name}/10")
    
    print(f"\n模态分布:")
    for modality, count in image_types.items():
        print(f"  {modality}: {count}/10")
    
    # 检查是否兼容
    compatible = has_file_name > 0
    extra_info = has_rgb_file_name > 0 or has_event_file_name > 0
    
    print(f"\n兼容性检查:")
    if compatible:
        print(f"  ✅ 兼容标准 COCO 格式 (有 file_name 字段)")
    else:
        print(f"  ❌ 不兼容标准 COCO 格式 (缺少 file_name 字段)")
    
    if extra_info:
        print(f"  ✅ 支持 Fusion 额外信息 (有 rgb_file_name / event_file_name)")
    
    # 打印第一个图像的完整信息
    if images:
        print(f"\n第一个图像信息示例:")
        sample = images[0]
        for key, value in sample.items():
            if isinstance(value, (str, int, float)) or value is None:
                print(f"  {key}: {value}")
        
        # 检查图像路径是否可访问
        file_name = sample.get('file_name') or sample.get('rgb_file_name') or sample.get('event_file_name')
        if file_name:
            print(f"\n文件路径示例: {file_name}")
    
    return compatible


def main():
    # 测试 train 集
    test_files = [
        'datasets/fred_fusion_v2/annotations/instances_train.json',
        'datasets/fred_fusion/annotations/instances_train.json',
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if Path(test_file).exists():
            result = test_fusion_compatibility(test_file)
            all_passed = all_passed and result
            print("\n" + "="*70 + "\n")
        else:
            print(f"文件不存在，跳过: {test_file}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())