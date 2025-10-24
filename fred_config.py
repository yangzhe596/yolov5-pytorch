#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED 数据集路径配置
在这里统一配置 FRED 数据集的根目录
"""
import os

# FRED 数据集根目录
# 修改这里的路径为你的 FRED 数据集实际位置
FRED_ROOT = os.environ.get('FRED_ROOT', '/home/yz/datasets/fred')

# COCO 格式数据集根目录
COCO_ROOT = 'datasets/fred_coco'

def get_fred_root():
    """获取 FRED 数据集根目录"""
    return FRED_ROOT

def get_coco_root():
    """获取 COCO 数据集根目录"""
    return COCO_ROOT

def get_image_dir(modality='rgb'):
    """
    获取图像目录
    
    Args:
        modality: 'rgb' 或 'event'
    
    Returns:
        FRED 数据集根目录（因为 file_name 包含相对路径）
    """
    return FRED_ROOT

def get_annotation_path(modality='rgb', split='train'):
    """
    获取标注文件路径
    
    Args:
        modality: 'rgb' 或 'event'
        split: 'train', 'val', 或 'test'
    
    Returns:
        标注文件的完整路径
    """
    return os.path.join(COCO_ROOT, modality, 'annotations', f'instances_{split}.json')

# 使用示例
if __name__ == '__main__':
    print(f"FRED 根目录: {get_fred_root()}")
    print(f"COCO 根目录: {get_coco_root()}")
    print(f"RGB 图像目录: {get_image_dir('rgb')}")
    print(f"Event 图像目录: {get_image_dir('event')}")
    print(f"RGB 训练集标注: {get_annotation_path('rgb', 'train')}")
    print(f"Event 测试集标注: {get_annotation_path('event', 'test')}")
