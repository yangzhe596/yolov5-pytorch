#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 FRED 数据集路径配置是否正确
"""
import os
import json
from pathlib import Path

def test_path_config(modality='rgb'):
    """测试路径配置"""
    
    print(f"\n{'='*70}")
    print(f"测试 FRED 数据集路径配置 - {modality.upper()} 模态")
    print(f"{'='*70}\n")
    
    # 配置
    fred_root = '/home/yz/datasets/fred'
    coco_root = f'datasets/fred_coco/{modality}'
    
    # 检查 FRED 根目录
    print(f"1. 检查 FRED 根目录: {fred_root}")
    if not os.path.exists(fred_root):
        print(f"   ❌ FRED 根目录不存在！")
        print(f"   请修改 fred_root 为正确的路径")
        return False
    print(f"   ✅ FRED 根目录存在\n")
    
    # 检查 COCO 标注文件
    ann_file = f'{coco_root}/annotations/instances_train.json'
    print(f"2. 检查 COCO 标注文件: {ann_file}")
    if not os.path.exists(ann_file):
        print(f"   ❌ COCO 标注文件不存在！")
        print(f"   请先运行: python convert_fred_to_coco.py --modality {modality}")
        return False
    print(f"   ✅ COCO 标注文件存在\n")
    
    # 加载标注文件
    print(f"3. 加载 COCO 标注文件...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data['images'])
    print(f"   ✅ 加载成功，共 {num_images} 张图片\n")
    
    # 检查前 5 张图片的路径
    print(f"4. 检查图片路径（前 5 张）:")
    success_count = 0
    fail_count = 0
    
    for i, img_info in enumerate(coco_data['images'][:5]):
        file_name = img_info['file_name']
        img_path = os.path.join(fred_root, file_name)
        
        exists = os.path.exists(img_path)
        status = "✅" if exists else "❌"
        
        print(f"   {status} [{i+1}] {file_name}")
        print(f"       完整路径: {img_path}")
        
        if exists:
            success_count += 1
        else:
            fail_count += 1
    
    print()
    
    # 统计所有图片
    print(f"5. 检查所有图片路径...")
    total_success = 0
    total_fail = 0
    missing_files = []
    
    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        img_path = os.path.join(fred_root, file_name)
        
        if os.path.exists(img_path):
            total_success += 1
        else:
            total_fail += 1
            if len(missing_files) < 10:  # 只记录前 10 个缺失文件
                missing_files.append(file_name)
    
    print(f"   总计: {num_images} 张图片")
    print(f"   ✅ 存在: {total_success} 张 ({total_success/num_images*100:.2f}%)")
    print(f"   ❌ 缺失: {total_fail} 张 ({total_fail/num_images*100:.2f}%)")
    
    if missing_files:
        print(f"\n   缺失文件示例（前 10 个）:")
        for f in missing_files:
            print(f"     - {f}")
    
    print()
    
    # 总结
    print(f"{'='*70}")
    if total_fail == 0:
        print(f"✅ 测试通过！所有图片路径都正确")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"❌ 测试失败！有 {total_fail} 张图片路径不正确")
        print(f"{'='*70}\n")
        print(f"可能的原因：")
        print(f"1. fred_root 路径配置错误")
        print(f"2. FRED 数据集结构不正确")
        print(f"3. COCO 标注文件中的 file_name 格式错误")
        print()
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 FRED 数据集路径配置')
    parser.add_argument('--modality', type=str, default='rgb', 
                        choices=['rgb', 'event'],
                        help='模态: rgb 或 event')
    args = parser.parse_args()
    
    success = test_path_config(args.modality)
    
    if success:
        print("✅ 可以开始训练了！")
        print(f"   python train_fred.py --modality {args.modality}")
    else:
        print("❌ 请先解决上述问题")
