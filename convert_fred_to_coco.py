#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将FRED数据集从YOLO格式转换为COCO格式
支持RGB和Event两种模态
"""
import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# 配置
FRED_ROOT = "/mnt/data/datasets/fred"
OUTPUT_ROOT = "datasets/fred_coco"
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.2    # 验证集比例
TEST_RATIO = 0.1   # 测试集比例

# 类别定义
CATEGORIES = [
    {"id": 1, "name": "object", "supercategory": "none"}
]

def create_coco_structure(output_root, modality):
    """创建COCO数据集目录结构"""
    coco_root = Path(output_root) / modality
    
    # 创建必要的目录
    dirs = [
        coco_root / "annotations",
        coco_root / "train",
        coco_root / "val",
        coco_root / "test",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    return coco_root

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    将YOLO格式边界框转换为COCO格式
    YOLO: (x_center, y_center, width, height) - 归一化坐标
    COCO: (x, y, width, height) - 像素坐标，其中(x,y)是左上角
    """
    x_center, y_center, width, height = yolo_bbox
    
    # 转换为像素坐标
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # 计算左上角坐标
    x = x_center_px - width_px / 2
    y = y_center_px - height_px / 2
    
    # 确保坐标在图像范围内
    x = max(0, x)
    y = max(0, y)
    width_px = min(width_px, img_width - x)
    height_px = min(height_px, img_height - y)
    
    return [x, y, width_px, height_px]

def collect_dataset_files(fred_root, modality):
    """
    收集所有有效的图片和标注文件对
    modality: 'rgb' 或 'event'
    """
    file_pairs = []
    fred_path = Path(fred_root)
    
    # 根据模态选择不同的目录
    if modality.lower() == 'rgb':
        img_subdir = "RGB"
        label_subdir = "RGB_YOLO"
        img_ext = ".jpg"
    elif modality.lower() == 'event':
        img_subdir = "Event/Frames"
        label_subdir = "Event_YOLO"
        img_ext = ".png"
    else:
        raise ValueError(f"不支持的模态: {modality}")
    
    # 遍历所有子目录（0-10）
    for video_dir in sorted(fred_path.iterdir()):
        if not video_dir.is_dir() or video_dir.name.startswith('.'):
            continue
        
        img_dir = video_dir / img_subdir
        yolo_dir = video_dir / label_subdir
        
        if not img_dir.exists() or not yolo_dir.exists():
            print(f"  跳过 {video_dir.name}: 缺少{img_subdir}或{label_subdir}目录")
            continue
        
        # 收集该视频序列的所有图片
        pattern = f"*{img_ext}"
        for img_file in img_dir.glob(pattern):
            label_file = yolo_dir / (img_file.stem + ".txt")
            
            # 只收集有标注的图片（标注文件存在且非空）
            if label_file.exists() and label_file.stat().st_size > 0:
                file_pairs.append({
                    'image': str(img_file),
                    'label': str(label_file),
                    'video_id': video_dir.name,
                    'filename': img_file.name
                })
    
    return file_pairs

def split_dataset(file_pairs, train_ratio, val_ratio, test_ratio):
    """划分数据集为训练集、验证集和测试集"""
    random.shuffle(file_pairs)
    
    total = len(file_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = file_pairs[:train_end]
    val_set = file_pairs[train_end:val_end]
    test_set = file_pairs[val_end:]
    
    return {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

def create_coco_annotation(file_list, coco_root, split_name, categories, modality):
    """创建COCO格式的标注文件"""
    
    # 初始化COCO格式的数据结构
    coco_data = {
        "info": {
            "description": f"FRED Dataset - {modality.upper()} Modality",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "FRED",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 1
    annotation_id = 1
    
    # 处理每个文件
    for file_info in tqdm(file_list, desc=f"处理{split_name}集"):
        # 读取图片获取尺寸
        img = Image.open(file_info['image'])
        img_width, img_height = img.size
        
        # 生成新的文件名
        new_filename = f"{file_info['video_id']}_{Path(file_info['filename']).stem}{Path(file_info['filename']).suffix}"
        
        # 复制图片到对应的split目录
        dst_img = coco_root / split_name / new_filename
        shutil.copy2(file_info['image'], dst_img)
        
        # 添加图片信息
        image_info = {
            "id": image_id,
            "file_name": new_filename,
            "width": img_width,
            "height": img_height,
            "license": 1,
            "date_captured": ""
        }
        coco_data["images"].append(image_info)
        
        # 读取YOLO标注并转换
        with open(file_info['label'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]
                
                # 转换边界框
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                
                # 计算面积
                area = coco_bbox[2] * coco_bbox[3]
                
                # 添加标注信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,  # COCO的category_id从1开始
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    # 保存COCO标注文件
    annotation_file = coco_root / "annotations" / f"instances_{split_name}.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return coco_data

def convert_dataset(fred_root, output_root, modality, categories, train_ratio, val_ratio, test_ratio):
    """转换整个数据集"""
    print("=" * 70)
    print(f"FRED数据集转换为COCO格式 - {modality.upper()} 模态")
    print("=" * 70)
    
    # 创建COCO目录结构
    coco_root = create_coco_structure(output_root, modality)
    print(f"✓ 创建COCO目录结构: {coco_root}")
    
    # 收集所有文件
    print(f"\n正在收集{modality.upper()}数据集文件...")
    file_pairs = collect_dataset_files(fred_root, modality)
    print(f"✓ 找到 {len(file_pairs)} 个有效的图片-标注对")
    
    if len(file_pairs) == 0:
        print(f"错误：未找到任何有效的{modality}数据！")
        return None
    
    # 划分数据集
    print("\n正在划分数据集...")
    datasets = split_dataset(file_pairs, train_ratio, val_ratio, test_ratio)
    
    print(f"✓ 训练集: {len(datasets['train'])} 张")
    print(f"✓ 验证集: {len(datasets['val'])} 张")
    print(f"✓ 测试集: {len(datasets['test'])} 张")
    
    # 转换各个数据集
    stats = {}
    for split_name, file_list in datasets.items():
        print(f"\n正在转换 {split_name} 集...")
        coco_data = create_coco_annotation(file_list, coco_root, split_name, categories, modality)
        
        stats[split_name] = {
            'images': len(coco_data['images']),
            'annotations': len(coco_data['annotations'])
        }
        
        print(f"✓ {split_name} 集转换完成")
        print(f"  - 图片数: {stats[split_name]['images']}")
        print(f"  - 标注数: {stats[split_name]['annotations']}")
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("转换完成！")
    print("=" * 70)
    print(f"模态: {modality.upper()}")
    print(f"输出目录: {coco_root}")
    print(f"\n数据集统计:")
    print(f"  训练集: {stats['train']['images']} 张图片, {stats['train']['annotations']} 个标注")
    print(f"  验证集: {stats['val']['images']} 张图片, {stats['val']['annotations']} 个标注")
    print(f"  测试集: {stats['test']['images']} 张图片, {stats['test']['annotations']} 个标注")
    print(f"  总计: {sum(s['images'] for s in stats.values())} 张图片, {sum(s['annotations'] for s in stats.values())} 个标注")
    
    # 保存数据集信息
    info_file = coco_root / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"FRED Dataset - {modality.upper()} Modality\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"源数据路径: {fred_root}\n\n")
        f.write("数据集划分:\n")
        f.write(f"  训练集: {train_ratio*100:.0f}%\n")
        f.write(f"  验证集: {val_ratio*100:.0f}%\n")
        f.write(f"  测试集: {test_ratio*100:.0f}%\n\n")
        f.write("统计信息:\n")
        for split_name, stat in stats.items():
            f.write(f"  {split_name}: {stat['images']} 张图片, {stat['annotations']} 个标注\n")
        f.write(f"\n类别信息:\n")
        for cat in categories:
            f.write(f"  ID {cat['id']}: {cat['name']}\n")
    
    return coco_root

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将FRED数据集转换为COCO格式')
    parser.add_argument('--modality', type=str, required=True, choices=['rgb', 'event', 'both'],
                        help='选择模态: rgb, event, 或 both')
    parser.add_argument('--fred_root', type=str, default=FRED_ROOT,
                        help='FRED数据集根目录')
    parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT,
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"错误：训练集、验证集和测试集比例之和必须为1.0，当前为{total_ratio}")
        return
    
    # 转换数据集
    modalities = ['rgb', 'event'] if args.modality == 'both' else [args.modality]
    
    for modality in modalities:
        print("\n")
        coco_root = convert_dataset(
            fred_root=args.fred_root,
            output_root=args.output_root,
            modality=modality,
            categories=CATEGORIES,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        if coco_root:
            print(f"\n{modality.upper()}模态数据集已保存到: {coco_root}")
    
    print("\n" + "=" * 70)
    print("所有转换完成！")
    print("=" * 70)
    print("\n下一步操作：")
    print("1. 查看生成的COCO标注文件: datasets/fred_coco/{modality}/annotations/")
    print("2. 使用COCO API加载和验证数据集")
    print("3. 根据需要修改训练脚本以支持COCO格式")

if __name__ == "__main__":
    main()
