#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建一个小型测试数据集
从FRED数据集中抽取少量样本用于快速测试
"""
import json
import os
import shutil
from pathlib import Path
import random

def create_mini_dataset(modality='rgb', num_train=50, num_val=20, num_test=10):
    """
    创建小型测试数据集
    
    Args:
        modality: 'rgb' 或 'event'
        num_train: 训练集样本数
        num_val: 验证集样本数
        num_test: 测试集样本数
    """
    print("=" * 80)
    print(f"创建小型测试数据集 - {modality.upper()}模态")
    print("=" * 80)
    
    # 源数据集路径
    src_root = f'datasets/fred_coco/{modality}'
    
    # 目标数据集路径
    dst_root = f'datasets/fred_mini/{modality}'
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dst_root, split), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'annotations'), exist_ok=True)
    
    # 处理每个split
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    for split, num_samples in splits.items():
        print(f"\n处理 {split} 集...")
        
        # 加载原始COCO标注
        src_json = os.path.join(src_root, 'annotations', f'instances_{split}.json')
        
        with open(src_json, 'r') as f:
            coco_data = json.load(f)
        
        # 随机选择样本
        random.seed(42)
        selected_images = random.sample(coco_data['images'], 
                                       min(num_samples, len(coco_data['images'])))
        
        # 获取选中图片的ID
        selected_img_ids = {img['id'] for img in selected_images}
        
        # 筛选对应的标注
        selected_annotations = [ann for ann in coco_data['annotations'] 
                               if ann['image_id'] in selected_img_ids]
        
        # 创建新的COCO数据
        mini_coco_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'images': selected_images,
            'annotations': selected_annotations,
            'categories': coco_data['categories']
        }
        
        # 保存新的标注文件
        dst_json = os.path.join(dst_root, 'annotations', f'instances_{split}.json')
        with open(dst_json, 'w') as f:
            json.dump(mini_coco_data, f, indent=2)
        
        # 复制图片
        src_img_dir = os.path.join(src_root, split)
        dst_img_dir = os.path.join(dst_root, split)
        
        for img_info in selected_images:
            src_img = os.path.join(src_img_dir, img_info['file_name'])
            dst_img = os.path.join(dst_img_dir, img_info['file_name'])
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
        
        print(f"  ✓ {split}: {len(selected_images)} 张图片, {len(selected_annotations)} 个标注")
    
    # 保存数据集信息
    info_file = os.path.join(dst_root, 'dataset_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"FRED Mini Dataset - {modality.upper()} Modality\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"用途: 快速测试训练流程\n")
        f.write(f"创建时间: {Path(src_json).stat().st_mtime}\n\n")
        f.write("数据集大小:\n")
        f.write(f"  训练集: {num_train} 张\n")
        f.write(f"  验证集: {num_val} 张\n")
        f.write(f"  测试集: {num_test} 张\n")
    
    print("\n" + "=" * 80)
    print("小型测试数据集创建完成！")
    print("=" * 80)
    print(f"位置: {dst_root}")
    print(f"总计: {num_train + num_val + num_test} 张图片")
    
    return dst_root

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='创建小型测试数据集')
    parser.add_argument('--modality', type=str, default='rgb', 
                       choices=['rgb', 'event', 'both'],
                       help='选择模态')
    parser.add_argument('--num_train', type=int, default=50,
                       help='训练集样本数')
    parser.add_argument('--num_val', type=int, default=20,
                       help='验证集样本数')
    parser.add_argument('--num_test', type=int, default=10,
                       help='测试集样本数')
    
    args = parser.parse_args()
    
    modalities = ['rgb', 'event'] if args.modality == 'both' else [args.modality]
    
    for mod in modalities:
        create_mini_dataset(mod, args.num_train, args.num_val, args.num_test)
        print()
    
    print("\n使用方法:")
    print("  /home/yz/miniforge3/envs/torch/bin/python3 test_train_mini.py --modality rgb")
