#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证COCO格式数据集的正确性
"""
import json
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_coco_annotation(json_path):
    """加载COCO标注文件"""
    with open(json_path, 'r') as f:
        return json.load(f)

def verify_coco_dataset(coco_root, split='train', show_samples=5):
    """验证COCO数据集"""
    print("=" * 70)
    print(f"验证COCO数据集: {coco_root} - {split}")
    print("=" * 70)
    
    # 加载标注文件
    ann_file = Path(coco_root) / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        print(f"错误：标注文件不存在: {ann_file}")
        return False
    
    print(f"\n✓ 加载标注文件: {ann_file}")
    coco_data = load_coco_annotation(ann_file)
    
    # 基本统计
    print(f"\n数据集统计:")
    print(f"  图片数量: {len(coco_data['images'])}")
    print(f"  标注数量: {len(coco_data['annotations'])}")
    print(f"  类别数量: {len(coco_data['categories'])}")
    
    # 类别信息
    print(f"\n类别信息:")
    for cat in coco_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        # 统计该类别的标注数量
        count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == cat_id)
        print(f"  ID {cat_id}: {cat_name} - {count} 个标注")
    
    # 验证图片文件
    print(f"\n验证图片文件...")
    img_dir = Path(coco_root) / split
    missing_images = []
    
    for img_info in coco_data['images']:
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"  ✗ 缺失 {len(missing_images)} 张图片")
        print(f"    示例: {missing_images[:5]}")
        return False
    else:
        print(f"  ✓ 所有图片文件都存在")
    
    # 验证标注
    print(f"\n验证标注...")
    invalid_anns = []
    
    for ann in coco_data['annotations']:
        bbox = ann['bbox']
        # 检查边界框格式
        if len(bbox) != 4:
            invalid_anns.append(f"标注{ann['id']}: 边界框格式错误")
            continue
        
        # 检查边界框值
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            invalid_anns.append(f"标注{ann['id']}: 边界框宽高无效 (w={w}, h={h})")
        
        # 检查面积
        if ann['area'] <= 0:
            invalid_anns.append(f"标注{ann['id']}: 面积无效 ({ann['area']})")
    
    if invalid_anns:
        print(f"  ✗ 发现 {len(invalid_anns)} 个无效标注")
        print(f"    示例: {invalid_anns[:5]}")
        return False
    else:
        print(f"  ✓ 所有标注都有效")
    
    # 统计信息
    print(f"\n详细统计:")
    
    # 每张图片的标注数量
    img_ann_count = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        img_ann_count[img_id] = img_ann_count.get(img_id, 0) + 1
    
    ann_counts = list(img_ann_count.values())
    print(f"  每张图片标注数:")
    print(f"    平均: {sum(ann_counts) / len(ann_counts):.2f}")
    print(f"    最小: {min(ann_counts)}")
    print(f"    最大: {max(ann_counts)}")
    
    # 边界框尺寸统计
    bbox_widths = [ann['bbox'][2] for ann in coco_data['annotations']]
    bbox_heights = [ann['bbox'][3] for ann in coco_data['annotations']]
    
    print(f"  边界框宽度 (像素):")
    print(f"    平均: {sum(bbox_widths) / len(bbox_widths):.2f}")
    print(f"    最小: {min(bbox_widths):.2f}")
    print(f"    最大: {max(bbox_widths):.2f}")
    
    print(f"  边界框高度 (像素):")
    print(f"    平均: {sum(bbox_heights) / len(bbox_heights):.2f}")
    print(f"    最小: {min(bbox_heights):.2f}")
    print(f"    最大: {max(bbox_heights):.2f}")
    
    # 可视化样本
    if show_samples > 0:
        print(f"\n可视化 {show_samples} 个样本...")
        visualize_samples(coco_data, img_dir, show_samples)
    
    print("\n" + "=" * 70)
    print("验证完成！数据集格式正确。")
    print("=" * 70)
    
    return True

def visualize_samples(coco_data, img_dir, num_samples=5):
    """可视化样本"""
    # 创建图片ID到标注的映射
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # 创建类别ID到名称的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 选择有标注的图片
    images_with_anns = [img for img in coco_data['images'] if img['id'] in img_to_anns]
    
    # 随机选择样本
    import random
    samples = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))
    
    # 创建子图
    fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 5))
    if len(samples) == 1:
        axes = [axes]
    
    for idx, img_info in enumerate(samples):
        # 加载图片
        img_path = img_dir / img_info['file_name']
        img = Image.open(img_path)
        
        # 显示图片
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Image {img_info['id']}\n{img_info['file_name']}", fontsize=8)
        
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
                cat_name,
                color='red',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(img_dir).parent / f"visualization_{Path(img_dir).name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 可视化结果已保存: {output_path}")
    
    # 显示图片
    try:
        plt.show()
    except:
        print("  注意: 无法显示图片（可能是在无GUI环境中运行）")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='验证COCO格式数据集')
    parser.add_argument('--modality', type=str, required=True, choices=['rgb', 'event'],
                        help='选择模态: rgb 或 event')
    parser.add_argument('--coco_root', type=str, default='datasets/fred_coco',
                        help='COCO数据集根目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='验证哪个数据集划分')
    parser.add_argument('--show_samples', type=int, default=5,
                        help='可视化样本数量（0表示不可视化）')
    
    args = parser.parse_args()
    
    # 构建完整路径
    coco_root = Path(args.coco_root) / args.modality
    
    if not coco_root.exists():
        print(f"错误：数据集目录不存在: {coco_root}")
        print(f"请先运行 convert_fred_to_coco.py 转换数据集")
        return
    
    # 验证数据集
    success = verify_coco_dataset(coco_root, args.split, args.show_samples)
    
    if success:
        print("\n✓ 数据集验证通过！")
    else:
        print("\n✗ 数据集验证失败！")

if __name__ == "__main__":
    main()
