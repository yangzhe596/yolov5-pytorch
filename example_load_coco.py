#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED COCO数据集加载示例
演示如何加载和使用转换后的COCO格式数据集
"""
import json
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_coco_data(coco_root, split='train'):
    """加载COCO标注数据"""
    ann_file = Path(coco_root) / "annotations" / f"instances_{split}.json"
    
    print(f"加载标注文件: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    return coco_data

def show_dataset_info(coco_data):
    """显示数据集基本信息"""
    print("\n" + "=" * 70)
    print("数据集信息")
    print("=" * 70)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  图片数量: {len(coco_data['images'])}")
    print(f"  标注数量: {len(coco_data['annotations'])}")
    print(f"  类别数量: {len(coco_data['categories'])}")
    
    # 类别信息
    print(f"\n类别信息:")
    for cat in coco_data['categories']:
        # 统计该类别的标注数
        count = sum(1 for ann in coco_data['annotations'] 
                   if ann['category_id'] == cat['id'])
        print(f"  ID {cat['id']}: {cat['name']} ({count} 个标注)")
    
    # 边界框统计
    if coco_data['annotations']:
        widths = [ann['bbox'][2] for ann in coco_data['annotations']]
        heights = [ann['bbox'][3] for ann in coco_data['annotations']]
        areas = [ann['area'] for ann in coco_data['annotations']]
        
        print(f"\n边界框统计:")
        print(f"  宽度: 平均={sum(widths)/len(widths):.2f}, "
              f"最小={min(widths):.2f}, 最大={max(widths):.2f}")
        print(f"  高度: 平均={sum(heights)/len(heights):.2f}, "
              f"最小={min(heights):.2f}, 最大={max(heights):.2f}")
        print(f"  面积: 平均={sum(areas)/len(areas):.2f}, "
              f"最小={min(areas):.2f}, 最大={max(areas):.2f}")

def visualize_samples(coco_data, coco_root, split, num_samples=3):
    """可视化样本"""
    print(f"\n可视化 {num_samples} 个样本...")
    
    # 创建image_id到annotations的映射
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # 创建category_id到name的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 选择有标注的图片
    images_with_anns = [img for img in coco_data['images'] 
                       if img['id'] in img_to_anns]
    
    if not images_with_anns:
        print("没有找到带标注的图片！")
        return
    
    # 随机选择样本
    import random
    samples = random.sample(images_with_anns, 
                          min(num_samples, len(images_with_anns)))
    
    # 创建子图
    fig, axes = plt.subplots(1, len(samples), figsize=(6*len(samples), 6))
    if len(samples) == 1:
        axes = [axes]
    
    for idx, img_info in enumerate(samples):
        # 加载图片
        img_path = Path(coco_root) / split / img_info['file_name']
        
        if not img_path.exists():
            print(f"警告: 图片不存在 {img_path}")
            continue
        
        img = Image.open(img_path)
        
        # 显示图片
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(
            f"ID: {img_info['id']}\n{img_info['file_name'][:30]}...",
            fontsize=10
        )
        
        # 绘制边界框
        anns = img_to_anns[img_info['id']]
        for ann in anns:
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # 创建矩形
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            axes[idx].add_patch(rect)
            
            # 添加类别标签
            cat_name = cat_id_to_name[ann['category_id']]
            axes[idx].text(
                x, y - 5,
                f"{cat_name}",
                color='red',
                fontsize=12,
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, 
                         edgecolor='none', pad=2)
            )
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(coco_root) / f"visualization_{split}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {output_path}")
    
    # 尝试显示
    try:
        plt.show()
    except:
        print("注意: 无法显示图片（可能在无GUI环境中运行）")
    
    plt.close()

def show_sample_annotation(coco_data, coco_root, split):
    """显示一个样本的详细标注信息"""
    print("\n" + "=" * 70)
    print("样本标注示例")
    print("=" * 70)
    
    # 找到第一个有标注的图片
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    if not img_to_anns:
        print("没有找到标注！")
        return
    
    # 获取第一个样本
    img_id = list(img_to_anns.keys())[0]
    img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
    anns = img_to_anns[img_id]
    
    print(f"\n图片信息:")
    print(f"  ID: {img_info['id']}")
    print(f"  文件名: {img_info['file_name']}")
    print(f"  尺寸: {img_info['width']} x {img_info['height']}")
    
    print(f"\n标注信息 ({len(anns)} 个):")
    for i, ann in enumerate(anns, 1):
        cat = next(c for c in coco_data['categories'] 
                  if c['id'] == ann['category_id'])
        bbox = ann['bbox']
        print(f"  标注 {i}:")
        print(f"    类别: {cat['name']} (ID: {cat['id']})")
        print(f"    边界框: x={bbox[0]:.2f}, y={bbox[1]:.2f}, "
              f"w={bbox[2]:.2f}, h={bbox[3]:.2f}")
        print(f"    面积: {ann['area']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='FRED COCO数据集加载示例')
    parser.add_argument('--modality', type=str, default='rgb',
                       choices=['rgb', 'event'],
                       help='选择模态')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='选择数据集划分')
    parser.add_argument('--coco_root', type=str, 
                       default='datasets/fred_coco',
                       help='COCO数据集根目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化样本')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 构建完整路径
    coco_root = Path(args.coco_root) / args.modality
    
    if not coco_root.exists():
        print(f"错误: 数据集目录不存在: {coco_root}")
        print("请先运行 convert_fred_to_coco.py 转换数据集")
        return
    
    print("=" * 70)
    print(f"FRED COCO数据集加载示例 - {args.modality.upper()} 模态")
    print("=" * 70)
    
    # 加载数据
    coco_data = load_coco_data(coco_root, args.split)
    
    # 显示信息
    show_dataset_info(coco_data)
    
    # 显示样本标注
    show_sample_annotation(coco_data, coco_root, args.split)
    
    # 可视化
    if args.visualize:
        visualize_samples(coco_data, coco_root, args.split, args.num_samples)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    
    # 使用提示
    print("\n使用提示:")
    print("1. 查看其他划分:")
    print(f"   python example_load_coco.py --modality {args.modality} --split val")
    print("2. 可视化样本:")
    print(f"   python example_load_coco.py --modality {args.modality} --visualize")
    print("3. 查看Event模态:")
    print(f"   python example_load_coco.py --modality event --split train")

if __name__ == "__main__":
    main()
