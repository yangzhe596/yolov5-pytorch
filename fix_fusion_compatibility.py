#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 Fusion 数据集的兼容性问题

为现有的 Fusion 数据集添加 file_name 字段，使其兼容标准 COCO 格式
"""

import json
from pathlib import Path


def fix_fusion_compatibility(json_path):
    """修复 Fusion 数据集的兼容性"""
    print(f"修复 Fusion 数据集: {json_path}")
    
    if not Path(json_path).exists():
        print(f"  ❌ 文件不存在")
        return False
    
    # 备份原文件
    backup_path = json_path.replace('.json', '_backup.json')
    Path(json_path).rename(backup_path)
    print(f"  ✓ 备份到: {backup_path}")
    
    # 加载备份文件
    with open(backup_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 修复图像数据
    images = coco_data.get('images', [])
    fixed_count = 0
    
    for img in images:
        # 如果没有 file_name 字段，根据 modality 选择
        if 'file_name' not in img:
            if img.get('modality') == 'dual' or 'rgb_file_name' in img:
                # 双模态或有 RGB：使用 RGB 作为主 file_name
                if 'rgb_file_name' in img:
                    img['file_name'] = img['rgb_file_name']
                    fixed_count += 1
            elif img.get('modality') == 'rgb':
                # 仅 RGB
                if 'rgb_file_name' in img:
                    img['file_name'] = img['rgb_file_name']
                    fixed_count += 1
            elif img.get('modality') == 'event' or 'event_file_name' in img:
                # 仅 Event
                if 'event_file_name' in img:
                    img['file_name'] = img['event_file_name']
                    fixed_count += 1
    
    # 保存修复后的文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 修复完成: {fixed_count}/{len(images)} 张图像")
    print(f"  ✓ 文件已更新: {json_path}")
    
    return True


def main():
    # 需要修复的文件
    fix_files = [
        'datasets/fred_fusion/annotations/instances_train.json',
        'datasets/fred_fusion/annotations/instances_val.json',
        'datasets/fred_fusion/annotations/instances_test.json',
    ]
    
    all_fixed = True
    
    for file_path in fix_files:
        if Path(file_path).exists():
            result = fix_fusion_compatibility(file_path)
            all_fixed = all_fixed and result
            print()
        else:
            print(f"文件不存在，跳过: {file_path}\n")
    
    if all_fixed:
        print("="*70)
        print("✅ 所有 Fusion 数据集已修复兼容性问题")
        print("="*70)
    else:
        print("⚠️  部分文件修复失败")
    
    return 0 if all_fixed else 1


if __name__ == '__main__':
    exit(main())