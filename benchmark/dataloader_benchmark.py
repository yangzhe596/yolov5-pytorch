#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载性能详细分析
细化分析数据加载过程中各个步骤的耗时：
- 图像读取
- 图像解码
- 数据增强 (Mosaic/MixUp/随机变换)
- 预处理
- 标签处理

使用方法:
    python benchmark/dataloader_benchmark.py --modality rgb --num_samples 100
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_fred as cfg
from utils.utils import cvtColor, preprocess_input


class TimingStats:
    """计时统计工具"""
    def __init__(self):
        self.times = defaultdict(list)
        self.current_timers = {}
    
    def start(self, name):
        """开始计时"""
        self.current_timers[name] = time.time()
    
    def end(self, name):
        """结束计时并记录"""
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.times[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return 0
    
    def get_stats(self, name):
        """获取统计信息"""
        if name not in self.times or len(self.times[name]) == 0:
            return None
        
        times = np.array(self.times[name])
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'total': np.sum(times),
            'count': len(times)
        }
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 90)
        print("数据加载性能详细分析")
        print("=" * 90)
        
        # 按平均时间排序
        sorted_names = sorted(self.times.keys(), 
                            key=lambda x: np.mean(self.times[x]), 
                            reverse=True)
        
        print(f"\n{'步骤':<25} {'平均(ms)':<12} {'标准差(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'总计(s)':<12}")
        print("-" * 90)
        
        total_time = 0
        for name in sorted_names:
            stats = self.get_stats(name)
            if stats:
                print(f"{name:<25} {stats['mean']*1000:>10.2f}  {stats['std']*1000:>10.2f}  "
                      f"{stats['min']*1000:>10.2f}  {stats['max']*1000:>10.2f}  {stats['total']:>10.2f}")
                total_time += stats['total']
        
        print("-" * 90)
        print(f"{'总计':<25} {'':<12} {'':<12} {'':<12} {'':<12} {total_time:>10.2f}")
        
        # 计算百分比
        print(f"\n{'步骤':<25} {'占比(%)':<12} {'吞吐量(samples/s)':<20}")
        print("-" * 90)
        
        for name in sorted_names:
            stats = self.get_stats(name)
            if stats and total_time > 0:
                percentage = (stats['total'] / total_time) * 100
                throughput = stats['count'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{name:<25} {percentage:>10.2f}  {throughput:>18.2f}")


def load_coco_annotations(coco_json_path, image_dir):
    """加载COCO标注"""
    print(f"加载COCO标注: {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像ID到文件名的映射
    image_id_to_info = {}
    for img in coco_data['images']:
        image_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # 构建图像ID到标注的映射
    image_id_to_annos = defaultdict(list)
    for anno in coco_data['annotations']:
        image_id_to_annos[anno['image_id']].append(anno)
    
    # 构建图像信息列表
    image_infos = []
    for img_id, img_info in image_id_to_info.items():
        # 构建完整路径
        if os.path.isabs(img_info['file_name']):
            # 如果是绝对路径，直接使用
            image_path = img_info['file_name']
        else:
            # 如果是相对路径，拼接 image_dir
            image_path = os.path.join(image_dir, img_info['file_name'])
        
        # 获取边界框
        boxes = []
        for anno in image_id_to_annos[img_id]:
            x, y, w, h = anno['bbox']
            # COCO格式: [x, y, width, height] -> [x1, y1, x2, y2, class_id]
            boxes.append([x, y, x + w, y + h, anno['category_id'] - 1])  # category_id从1开始
        
        image_infos.append({
            'path': image_path,
            'width': img_info['width'],
            'height': img_info['height'],
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
        })
    
    print(f"✓ 加载完成: {len(image_infos)} 张图片")
    return image_infos


def benchmark_image_loading(image_infos, num_samples, stats):
    """测试图像读取和解码"""
    print(f"\n[1/5] 测试图像读取和解码 ({num_samples} 张)...")
    
    indices = np.random.choice(len(image_infos), min(num_samples, len(image_infos)), replace=False)
    
    for idx in tqdm(indices, desc="  进度"):
        info = image_infos[idx]
        
        # 测试 PIL 读取
        stats.start("1_image_read_pil")
        image = Image.open(info['path'])
        stats.end("1_image_read_pil")
        
        # 测试 PIL 转换
        stats.start("2_image_convert_rgb")
        image = cvtColor(image)
        stats.end("2_image_convert_rgb")


def benchmark_resize_operations(image_infos, num_samples, input_shape, stats):
    """测试图像缩放操作"""
    print(f"\n[2/5] 测试图像缩放操作 ({num_samples} 张)...")
    
    indices = np.random.choice(len(image_infos), min(num_samples, len(image_infos)), replace=False)
    
    for idx in tqdm(indices, desc="  进度"):
        info = image_infos[idx]
        
        # 读取图像
        image = Image.open(info['path'])
        image = cvtColor(image)
        
        iw, ih = image.size
        h, w = input_shape
        
        # 测试简单缩放（无数据增强）
        stats.start("3_resize_simple")
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        
        image_resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image_resized, (dx, dy))
        stats.end("3_resize_simple")
        
        # 测试随机缩放（数据增强）
        stats.start("4_resize_augmented")
        jitter = 0.3
        new_ar = iw/ih * np.random.uniform(1-jitter, 1+jitter) / np.random.uniform(1-jitter, 1+jitter)
        scale = np.random.uniform(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image_aug = image.resize((nw, nh), Image.BICUBIC)
        
        dx = int(np.random.uniform(0, w-nw))
        dy = int(np.random.uniform(0, h-nh))
        new_image_aug = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_aug.paste(image_aug, (dx, dy))
        stats.end("4_resize_augmented")


def benchmark_color_augmentation(image_infos, num_samples, input_shape, stats):
    """测试颜色增强"""
    print(f"\n[3/5] 测试颜色增强 ({num_samples} 张)...")
    
    indices = np.random.choice(len(image_infos), min(num_samples, len(image_infos)), replace=False)
    
    for idx in tqdm(indices, desc="  进度"):
        info = image_infos[idx]
        
        # 读取并缩放图像
        image = Image.open(info['path'])
        image = cvtColor(image)
        
        iw, ih = image.size
        h, w = input_shape
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        
        # 测试颜色增强
        stats.start("5_color_augmentation")
        image_data = np.array(new_image, np.uint8)
        
        # HSV 颜色空间变换
        hue = 0.1
        sat = 0.7
        val = 0.4
        
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        stats.end("5_color_augmentation")


def benchmark_preprocessing(image_infos, num_samples, input_shape, stats):
    """测试预处理"""
    print(f"\n[4/5] 测试预处理 ({num_samples} 张)...")
    
    indices = np.random.choice(len(image_infos), min(num_samples, len(image_infos)), replace=False)
    
    for idx in tqdm(indices, desc="  进度"):
        info = image_infos[idx]
        
        # 读取并缩放图像
        image = Image.open(info['path'])
        image = cvtColor(image)
        
        iw, ih = image.size
        h, w = input_shape
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        
        # 测试预处理
        stats.start("6_preprocessing")
        image_array = np.array(new_image, dtype=np.float32)
        image_preprocessed = preprocess_input(image_array)
        image_transposed = np.transpose(image_preprocessed, (2, 0, 1))
        stats.end("6_preprocessing")


def benchmark_bbox_processing(image_infos, num_samples, input_shape, stats):
    """测试边界框处理"""
    print(f"\n[5/5] 测试边界框处理 ({num_samples} 张)...")
    
    indices = np.random.choice(len(image_infos), min(num_samples, len(image_infos)), replace=False)
    
    for idx in tqdm(indices, desc="  进度"):
        info = image_infos[idx]
        
        if len(info['boxes']) == 0:
            continue
        
        box = info['boxes'].copy()
        
        # 测试边界框变换
        stats.start("7_bbox_transform")
        iw, ih = info['width'], info['height']
        h, w = input_shape
        
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        
        # 调整边界框
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)]
        
        # 归一化
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        stats.end("7_bbox_transform")


def benchmark_mosaic_augmentation(image_infos, num_samples, input_shape, stats):
    """测试 Mosaic 数据增强"""
    print(f"\n[额外] 测试 Mosaic 数据增强 ({num_samples//4} 组)...")
    
    num_mosaic = num_samples // 4
    
    for _ in tqdm(range(num_mosaic), desc="  进度"):
        stats.start("8_mosaic_augmentation")
        
        # 随机选择4张图片
        indices = np.random.choice(len(image_infos), 4, replace=False)
        
        # 读取4张图片
        images = []
        for idx in indices:
            info = image_infos[idx]
            image = Image.open(info['path'])
            image = cvtColor(image)
            images.append(image)
        
        # 简化的 Mosaic 拼接
        h, w = input_shape
        min_offset_x = 0.3
        min_offset_y = 0.3
        
        cut_x = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
        cut_y = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))
        
        new_image = np.zeros([h, w, 3], dtype=np.uint8)
        
        # 左上
        img = images[0].resize((cut_x, cut_y), Image.BICUBIC)
        new_image[:cut_y, :cut_x] = np.array(img)
        
        # 右上
        img = images[1].resize((w-cut_x, cut_y), Image.BICUBIC)
        new_image[:cut_y, cut_x:] = np.array(img)
        
        # 左下
        img = images[2].resize((cut_x, h-cut_y), Image.BICUBIC)
        new_image[cut_y:, :cut_x] = np.array(img)
        
        # 右下
        img = images[3].resize((w-cut_x, h-cut_y), Image.BICUBIC)
        new_image[cut_y:, cut_x:] = np.array(img)
        
        stats.end("8_mosaic_augmentation")


def main():
    parser = argparse.ArgumentParser(description='数据加载性能详细分析')
    
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                       help='选择模态: rgb 或 event')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='测试样本数量 (默认: 100)')
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("数据加载性能详细分析")
    print("=" * 90)
    print(f"模态: {args.modality}")
    print(f"测试样本数: {args.num_samples}")
    print(f"输入尺寸: {cfg.INPUT_SHAPE}")
    print("=" * 90)
    
    # 创建计时器
    stats = TimingStats()
    
    # 加载数据集
    train_json = cfg.get_annotation_path(args.modality, 'train')
    train_img_dir = cfg.get_image_dir(args.modality)
    
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"训练集标注文件不存在: {train_json}")
    
    image_infos = load_coco_annotations(train_json, train_img_dir)
    
    # 运行各项测试
    benchmark_image_loading(image_infos, args.num_samples, stats)
    benchmark_resize_operations(image_infos, args.num_samples, cfg.INPUT_SHAPE, stats)
    benchmark_color_augmentation(image_infos, args.num_samples, cfg.INPUT_SHAPE, stats)
    benchmark_preprocessing(image_infos, args.num_samples, cfg.INPUT_SHAPE, stats)
    benchmark_bbox_processing(image_infos, args.num_samples, cfg.INPUT_SHAPE, stats)
    benchmark_mosaic_augmentation(image_infos, args.num_samples, cfg.INPUT_SHAPE, stats)
    
    # 打印结果
    stats.print_summary()
    
    # 详细分析
    print("\n" + "=" * 90)
    print("详细分析")
    print("=" * 90)
    
    total_time = sum(stats.get_stats(key)['total'] for key in stats.times.keys() if stats.get_stats(key))
    
    # 分组统计
    groups = {
        '图像I/O': ['1_image_read_pil', '2_image_convert_rgb'],
        '图像缩放': ['3_resize_simple', '4_resize_augmented'],
        '颜色增强': ['5_color_augmentation'],
        '预处理': ['6_preprocessing'],
        '边界框处理': ['7_bbox_transform'],
        'Mosaic增强': ['8_mosaic_augmentation']
    }
    
    print(f"\n{'分组':<20} {'总耗时(s)':<15} {'占比(%)':<12} {'平均单次(ms)':<15}")
    print("-" * 90)
    
    for group_name, keys in groups.items():
        group_time = 0
        group_count = 0
        for key in keys:
            stat = stats.get_stats(key)
            if stat:
                group_time += stat['total']
                group_count += stat['count']
        
        if group_time > 0:
            percentage = (group_time / total_time) * 100
            avg_time = (group_time / group_count * 1000) if group_count > 0 else 0
            print(f"{group_name:<20} {group_time:>13.2f}  {percentage:>10.2f}  {avg_time:>13.2f}")
    
    # 优化建议
    print("\n" + "=" * 90)
    print("优化建议")
    print("=" * 90)
    
    # 找出最耗时的步骤
    bottlenecks = []
    for key in stats.times.keys():
        stat = stats.get_stats(key)
        if stat:
            percentage = (stat['total'] / total_time) * 100
            bottlenecks.append((key, percentage, stat['mean']))
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    print("\n主要耗时步骤 (占比 > 10%):")
    for name, percentage, avg_time in bottlenecks[:5]:
        if percentage > 10:
            print(f"  - {name}: {percentage:.1f}% (平均 {avg_time*1000:.2f} ms)")
    
    # 具体建议
    print("\n具体优化建议:")
    
    for name, percentage, avg_time in bottlenecks[:3]:
        if '1_image_read_pil' in name and percentage > 20:
            print("  1. 图像读取是瓶颈:")
            print("     - 使用 SSD 存储数据集")
            print("     - 考虑使用 cv2.imread 替代 PIL")
            print("     - 预先将图像缓存到内存")
        
        if '4_resize_augmented' in name and percentage > 15:
            print("  2. 随机缩放耗时较高:")
            print("     - 减少 jitter 参数")
            print("     - 使用更快的插值方法 (BILINEAR 替代 BICUBIC)")
            print("     - 降低数据增强概率")
        
        if '5_color_augmentation' in name and percentage > 15:
            print("  3. 颜色增强耗时较高:")
            print("     - 减少 HSV 变换的强度")
            print("     - 降低颜色增强概率")
            print("     - 考虑使用 GPU 加速")
        
        if '8_mosaic_augmentation' in name and percentage > 20:
            print("  4. Mosaic 增强耗时较高:")
            print("     - 降低 Mosaic 概率")
            print("     - 减少 special_aug_ratio")
            print("     - 考虑在前期 epoch 禁用 Mosaic")
    
    print("\n" + "=" * 90)
    print("分析完成")
    print("=" * 90)


if __name__ == '__main__':
    main()
