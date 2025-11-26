#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED 数据集转换为 COCO 格式 - YOLOv5 兼容版本

基于 FRED 官方数据集生成指南，生成与 YOLOv5-PyTorch 完全兼容的 COCO 格式数据集。

主要改进：
1. 使用 interpolated_coordinates.txt（包含 drone_id，支持多目标追踪）
2. 支持帧级别划分（Frame-level Split）- 数据分布更均衡
3. 支持序列级别划分（Sequence-level Split）- 更好的泛化评估
4. 完整的数据验证和统计
5. 与 YOLOv5 训练脚本无缝集成

使用方法：
    # 帧级别划分（推荐）
    python convert_fred_to_coco_v2.py --split-mode frame --modality both
    
    # 序列级别划分
    python convert_fred_to_coco_v2.py --split-mode sequence --modality both
    
    # 仅转换 RGB 模态
    python convert_fred_to_coco_v2.py --modality rgb
    
    # 简化模式：随机抽取 10% 数据
    python convert_fred_to_coco_v2.py --simple
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
import hashlib
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FREDtoCOCOConverter:
    """FRED 数据集转换为 COCO 格式"""
    
    def __init__(self, fred_root, output_root, split_mode='frame', simple=False, simple_ratio=0.1):
        """
        初始化转换器
        
        Args:
            fred_root: FRED 数据集根目录
            output_root: 输出目录
            split_mode: 'frame' 或 'sequence'
            simple: 是否启用简化模式
            simple_ratio: 简化模式下的采样比例
        """
        self.fred_root = Path(fred_root)
        self.output_root = Path(output_root)
        self.split_mode = split_mode
        self.simple = simple
        self.simple_ratio = simple_ratio
        
        if not self.fred_root.exists():
            raise FileNotFoundError(f"FRED 数据集根目录不存在: {self.fred_root}")
        
        # COCO 类别定义
        self.categories = [
            {
                "id": 1,
                "name": "drone",
                "supercategory": "vehicle"
            }
        ]
        
        logger.info(f"FRED 根目录: {self.fred_root}")
        logger.info(f"输出目录: {self.output_root}")
        logger.info(f"划分模式: {split_mode}")
        if self.simple:
            logger.info(f"简化模式: 随机采样 {self.simple_ratio * 100:.1f}% 数据")
    
    def get_all_sequences(self):
        """获取所有可用的序列 ID"""
        sequences = []
        for seq_dir in sorted(self.fred_root.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                sequences.append(int(seq_dir.name))
        return sorted(sequences)
    
    def load_annotations(self, sequence_path):
        """
        加载标注文件（优先使用 interpolated_coordinates.txt）
        
        Args:
            sequence_path: 序列目录路径
            
        Returns:
            dict: {timestamp: [{'bbox': (x1,y1,x2,y2), 'drone_id': id}, ...]}
        """
        # 优先使用插值标注文件
        annotation_file = sequence_path / "interpolated_coordinates.txt"
        
        if not annotation_file.exists():
            logger.warning(f"未找到 interpolated_coordinates.txt，尝试使用 coordinates.txt")
            annotation_file = sequence_path / "coordinates.txt"
        
        if not annotation_file.exists():
            logger.warning(f"序列 {sequence_path.name} 无标注文件")
            return {}
        
        annotations = {}
        
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析格式: "时间: x1, y1, x2, y2, id" 或 "时间: x1, y1, x2, y2"
                    time_str, coords_str = line.split(': ')
                    timestamp = float(time_str)
                    
                    coords = [x.strip() for x in coords_str.split(',')]
                    
                    x1, y1, x2, y2 = map(float, coords[:4])
                    drone_id = int(float(coords[4])) if len(coords) > 4 else 1
                    
                    # 验证边界框有效性
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(
                            f"{annotation_file.name} 第 {line_num} 行: "
                            f"无效边界框 ({x1},{y1},{x2},{y2})"
                        )
                        continue
                    
                    if timestamp not in annotations:
                        annotations[timestamp] = []
                    
                    annotations[timestamp].append({
                        'bbox': (x1, y1, x2, y2),
                        'drone_id': drone_id
                    })
                    
                except Exception as e:
                    logger.warning(
                        f"{annotation_file.name} 第 {line_num} 行解析失败: {e}"
                    )
                    continue
        
        logger.info(
            f"序列 {sequence_path.name}: 加载 {len(annotations)} 个时间戳的标注"
        )
        return annotations
    
    def get_frames(self, sequence_path, modality):
        """
        获取帧列表及其时间戳
        
        Args:
            sequence_path: 序列目录路径
            modality: 'rgb' 或 'event'
            
        Returns:
            list: [(timestamp, frame_path), ...]
        """
        if modality == 'rgb':
            # 使用 PADDED_RGB（与 Event 对齐）
            frame_dir = sequence_path / "PADDED_RGB"
            if not frame_dir.exists():
                # 回退到原始 RGB
                frame_dir = sequence_path / "RGB"
            pattern = "*.jpg"
        elif modality == 'event':
            frame_dir = sequence_path / "Event" / "Frames"
            pattern = "*.png"
        else:
            raise ValueError(f"未知模态: {modality}")
        
        if not frame_dir.exists():
            logger.warning(f"帧目录不存在: {frame_dir}")
            return []
        
        frames = []
        
        for frame_path in sorted(frame_dir.glob(pattern)):
            timestamp = self._extract_timestamp(frame_path.name, modality)
            if timestamp is not None:
                frames.append((timestamp, frame_path))
        
        # 按时间戳排序
        frames = sorted(frames, key=lambda x: x[0])
        
        # 转换为相对时间戳
        if frames:
            first_timestamp = frames[0][0]
            frames = [(t - first_timestamp, path) for t, path in frames]
        
        return frames
    
    def _extract_timestamp(self, filename, modality):
        """从文件名提取时间戳"""
        try:
            if modality == 'rgb':
                # Video_0_16_03_03.363444.jpg
                name = filename.replace('.jpg', '')
                parts = name.split('_')
                
                if len(parts) >= 4:
                    time_parts = parts[-3:]
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = float(time_parts[2])
                    
                    return hours * 3600 + minutes * 60 + seconds
            
            elif modality == 'event':
                # Video_0_frame_100032333.png
                name = filename.replace('.png', '')
                parts = name.split('_')
                
                if len(parts) >= 3:
                    timestamp_us = int(parts[-1])
                    return timestamp_us / 1_000_000
        
        except Exception as e:
            logger.warning(f"无法从文件名 '{filename}' 提取时间戳: {e}")
        
        return None
    
    def find_closest_annotation(self, timestamp, annotations, threshold=0.05):
        """
        查找最接近的标注
        
        Args:
            timestamp: 目标时间戳
            annotations: 标注字典
            threshold: 时间容差（秒）
            
        Returns:
            list: 标注列表或空列表
        """
        if not annotations:
            return []
        
        closest_time = min(annotations.keys(), key=lambda t: abs(t - timestamp))
        
        if abs(closest_time - timestamp) <= threshold:
            return annotations[closest_time]
        
        return []
    
    def validate_bbox(self, bbox, width, height):
        """
        验证并修正边界框
        
        Args:
            bbox: (x1, y1, x2, y2)
            width: 图像宽度
            height: 图像高度
            
        Returns:
            tuple: (is_valid, corrected_bbox)
        """
        x1, y1, x2, y2 = bbox
        
        # 确保坐标顺序正确
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        
        # 限制在图像边界内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 确保有效面积
        if x2 <= x1 or y2 <= y1:
            return False, None
        
        return True, (x1, y1, x2, y2)
    
    def _get_relative_path(self, sequence_id, frame_path):
        """生成相对于 FRED 根目录的相对路径"""
        try:
            relative_path = frame_path.relative_to(self.fred_root)
            return str(relative_path)
        except ValueError:
            return f"{sequence_id}/{frame_path.name}"
    
    def process_sequence(self, sequence_id, modality, image_id_offset=0, 
                        annotation_id_offset=0):
        """
        处理单个序列
        
        Args:
            sequence_id: 序列 ID
            modality: 'rgb' 或 'event'
            image_id_offset: 图像 ID 起始值
            annotation_id_offset: 标注 ID 起始值
            
        Returns:
            tuple: (images, annotations, next_image_id, next_annotation_id, stats)
        """
        sequence_path = self.fred_root / str(sequence_id)
        
        if not sequence_path.exists():
            logger.warning(f"序列 {sequence_id} 不存在")
            return [], [], image_id_offset, annotation_id_offset, {}
        
        # 加载标注
        annotations_dict = self.load_annotations(sequence_path)
        
        # 获取帧
        frames = self.get_frames(sequence_path, modality)
        
        if not frames:
            logger.warning(f"序列 {sequence_id} ({modality}) 无帧")
            return [], [], image_id_offset, annotation_id_offset, {}
        
        images = []
        annotations = []
        
        image_id = image_id_offset
        annotation_id = annotation_id_offset
        
        # 统计信息
        stats = {
            'total_frames': len(frames),
            'matched_frames': 0,
            'total_annotations': 0,
            'invalid_bboxes': 0
        }
        
        # 图像尺寸（FRED 数据集固定尺寸）
        width, height = 1280, 720
        
        for idx, (timestamp, frame_path) in enumerate(frames):
            # 简化模式下过滤非采样帧
            if self.simple and self.sampled_frames is not None:
                key = (sequence_id, modality, idx, timestamp)
                if key not in self.sampled_frames:
                    continue
                # 标记帧已使用，避免重复计数
                self.sampled_frames.remove(key)
            
            # 查找匹配的标注
            anns = self.find_closest_annotation(timestamp, annotations_dict)
            
            # 仅包含有标注的帧
            if not anns:
                continue
            
            stats['matched_frames'] += 1
            
            image_entry = {
                "id": image_id,
                "file_name": self._get_relative_path(sequence_id, frame_path),
                "width": width,
                "height": height,
                "sequence_id": sequence_id,
                "timestamp": timestamp
            }
            images.append(image_entry)
            
            for ann in anns:
                bbox = ann['bbox']
                is_valid, corrected_bbox = self.validate_bbox(bbox, width, height)
                if not is_valid:
                    stats['invalid_bboxes'] += 1
                    continue
                x1, y1, x2, y2 = corrected_bbox
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox_coco,
                    "area": area,
                    "iscrowd": 0,
                    "drone_id": ann['drone_id']
                }
                annotations.append(annotation_entry)
                annotation_id += 1
                stats['total_annotations'] += 1
            
            image_id += 1
        
        return images, annotations, image_id, annotation_id, stats
    
    def split_sequences(self, sequences, train_ratio=0.7, val_ratio=0.15, 
                       test_ratio=0.15, seed=42):
        """
        序列级别划分
        
        Args:
            sequences: 序列 ID 列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            
        Returns:
            tuple: (train_seqs, val_seqs, test_seqs)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        random.seed(seed)
        sequences = sequences.copy()
        random.shuffle(sequences)
        
        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train:n_train + n_val]
        test_seqs = sequences[n_train + n_val:]
        
        return sorted(train_seqs), sorted(val_seqs), sorted(test_seqs)
    
    def get_frame_split(self, image_id, sequence_id, train_ratio=0.7, 
                       val_ratio=0.15, seed=42):
        """
        帧级别划分（确定性哈希）
        
        Args:
            image_id: 图像 ID
            sequence_id: 序列 ID
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            str: 'train', 'val', 或 'test'
        """
        hash_input = f"{sequence_id}_{image_id}_{seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        rand_val = (hash_value % 1000000) / 1000000.0
        
        if rand_val < train_ratio:
            return 'train'
        elif rand_val < train_ratio + val_ratio:
            return 'val'
        else:
            return 'test'
    
    def generate_coco_split(self, sequences, modality, split_name):
        """
        生成指定划分的 COCO 数据
        
        Args:
            sequences: 序列 ID 列表
            modality: 'rgb' 或 'event'
            split_name: 'train', 'val', 或 'test'
            
        Returns:
            dict: COCO 格式字典
        """
        logger.info(f"\n生成 {split_name} 划分 ({modality.upper()})...")
        logger.info(f"序列: {sequences}")
        
        all_images = []
        all_annotations = []
        
        image_id = 1
        annotation_id = 1
        
        total_stats = defaultdict(int)
        
        for seq_id in tqdm(sequences, desc=f"处理序列 ({split_name})"):
            images, annotations, image_id, annotation_id, stats = \
                self.process_sequence(seq_id, modality, image_id, annotation_id)
            
            all_images.extend(images)
            all_annotations.extend(annotations)
            
            for key, value in stats.items():
                total_stats[key] += value
        
        # 创建 COCO 格式字典
        coco_dict = {
            "info": {
                "description": f"FRED Dataset - {modality.upper()} - {split_name.upper()} split ({self.split_mode}-level)",
                "url": "https://github.com/miccunifi/FRED",
                "version": "2.0",
                "year": 2025,
                "contributor": "MICC - University of Florence",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Apache License 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0"
                }
            ],
            "categories": self.categories,
            "images": all_images,
            "annotations": all_annotations
        }
        
        logger.info(f"  图像数: {len(all_images)}")
        logger.info(f"  标注数: {len(all_annotations)}")
        logger.info(f"  序列数: {len(sequences)}")
        logger.info(f"  匹配率: {total_stats['matched_frames']}/{total_stats['total_frames']} "
                   f"({total_stats['matched_frames']/max(total_stats['total_frames'],1)*100:.1f}%)")
        
        return coco_dict, total_stats
    
    def generate_frame_split(self, modality, train_ratio=0.7, val_ratio=0.15, 
                            test_ratio=0.15, seed=42, sequences=None):
        """
        生成帧级别划分的 COCO 数据集
        
        Args:
            modality: 'rgb' 或 'event'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            sequences: 可选的序列列表（用于简化模式）
            
        Returns:
            dict: 包含 train/val/test 的字典
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"帧级别划分 - {modality.upper()} 模态")
        logger.info(f"{'='*70}")
        
        if sequences is None:
            sequences = self.get_all_sequences()
        else:
            sequences = sorted(sequences)
        
        logger.info(f"使用 {len(sequences)} 个序列生成帧级别划分")
        
        # 收集所有帧
        all_frames = {
            'train': {'images': [], 'annotations': []},
            'val': {'images': [], 'annotations': []},
            'test': {'images': [], 'annotations': []}
        }
        
        image_id = 1
        annotation_id = 1
        
        for seq_id in tqdm(sequences, desc="收集帧"):
            images, annotations, _, _, _ = \
                self.process_sequence(seq_id, modality, 0, 0)
            
            # 按帧划分
            for img, anns_for_img in zip(images, 
                                         self._group_annotations_by_image(annotations)):
                split = self.get_frame_split(img['id'], seq_id, 
                                            train_ratio, val_ratio, seed)
                
                # 重新分配 ID
                img['id'] = image_id
                all_frames[split]['images'].append(img)
                
                for ann in anns_for_img:
                    ann['id'] = annotation_id
                    ann['image_id'] = image_id
                    all_frames[split]['annotations'].append(ann)
                    annotation_id += 1
                
                image_id += 1
        
        # 生成 COCO 格式
        results = {}
        for split in ['train', 'val', 'test']:
            results[split] = {
                "info": {
                    "description": f"FRED Dataset - {modality.upper()} - {split.upper()} split (frame-level)",
                    "url": "https://github.com/miccunifi/FRED",
                    "version": "2.0",
                    "year": 2025,
                    "contributor": "MICC - University of Florence",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Apache License 2.0",
                        "url": "https://www.apache.org/licenses/LICENSE-2.0"
                    }
                ],
                "categories": self.categories,
                "images": all_frames[split]['images'],
                "annotations": all_frames[split]['annotations']
            }
            
            logger.info(f"{split}: {len(all_frames[split]['images'])} 图像, "
                       f"{len(all_frames[split]['annotations'])} 标注")
        
        return results
    
    def _group_annotations_by_image(self, annotations):
        """将标注按图像 ID 分组"""
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann['image_id']].append(ann)
        return [grouped[img_id] for img_id in sorted(grouped.keys())]
    
    def generate_all_splits(self, modality='both', train_ratio=0.7, 
                           val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        生成所有划分和模态的 COCO 数据集
        
        Args:
            modality: 'rgb', 'event', 或 'both'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FRED 转 COCO - {self.split_mode.upper()} 级别划分")
        logger.info(f"{'='*70}")
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        modalities = ['rgb', 'event'] if modality == 'both' else [modality]
        
        if self.simple:
            logger.info("启用简化模式: 随机采样帧数据")
            random.seed(seed)
        
        # 预扫描选择所有帧（用于简化模式）
        self.sampled_frames = None
        if self.simple:
            logger.info("统计所有帧用以采样 10%")
            frame_index = []  # 存储 (sequence_id, modality, frame_idx_in_sequence, timestamp)
            for seq_id in self.get_all_sequences():
                sequence_path = self.fred_root / str(seq_id)
                annotations_dict = self.load_annotations(sequence_path)
                if not annotations_dict:
                    continue
                for modality_candidate in ('rgb', 'event'):
                    frames = self.get_frames(sequence_path, modality_candidate)
                    for idx, (timestamp, _) in enumerate(frames):
                        anns = self.find_closest_annotation(timestamp, annotations_dict)
                        if anns:
                            frame_index.append((seq_id, modality_candidate, idx, timestamp))
            if not frame_index:
                raise ValueError("未找到任何带标注的帧，无法执行简化模式采样")
            sample_size = max(1, int(len(frame_index) * self.simple_ratio))
            self.sampled_frames = set(random.sample(frame_index, sample_size))
            logger.info(f"总可用帧: {len(frame_index)}, 采样帧数: {len(self.sampled_frames)}")
        
        for mod in modalities:
            if self.split_mode == 'sequence':
                sequences = self.get_all_sequences()
                train_seqs, val_seqs, test_seqs = self.split_sequences(
                    sequences, train_ratio, val_ratio, test_ratio, seed
                )
                
                logger.info(f"\n序列划分:")
                logger.info(f"  训练: {len(train_seqs)} 序列")
                logger.info(f"  验证: {len(val_seqs)} 序列")
                logger.info(f"  测试: {len(test_seqs)} 序列")
                
                for split_name, seqs in [('train', train_seqs), 
                                        ('val', val_seqs), 
                                        ('test', test_seqs)]:
                    coco_dict, stats = self.generate_coco_split(seqs, mod, split_name)
                    
                    output_file = self.output_root / mod / 'annotations' / \
                                 f'instances_{split_name}.json'
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'w') as f:
                        json.dump(coco_dict, f, indent=2)
                    
                    logger.info(f"✓ 已保存: {output_file}")
            
                train_seqs, val_seqs, test_seqs = self.split_sequences(
                    sequences, train_ratio, val_ratio, test_ratio, seed
                )
                
                logger.info(f"\n序列划分:")
                logger.info(f"  训练: {len(train_seqs)} 序列")
                logger.info(f"  验证: {len(val_seqs)} 序列")
                logger.info(f"  测试: {len(test_seqs)} 序列")
                
                for split_name, seqs in [('train', train_seqs), 
                                        ('val', val_seqs), 
                                        ('test', test_seqs)]:
                    coco_dict, stats = self.generate_coco_split(seqs, mod, split_name)
                    
                    output_file = self.output_root / mod / 'annotations' / \
                                 f'instances_{split_name}.json'
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'w') as f:
                        json.dump(coco_dict, f, indent=2)
                    
                    logger.info(f"✓ 已保存: {output_file}")
            
            else:  # frame-level
                results = self.generate_frame_split(
                    mod, train_ratio, val_ratio, test_ratio, seed,
                    sequences=None
                )
                
                for split_name, coco_dict in results.items():
                    output_file = self.output_root / mod / 'annotations' / \
                                 f'instances_{split_name}.json'
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'w') as f:
                        json.dump(coco_dict, f, indent=2)
                    
                    logger.info(f"✓ 已保存: {output_file}")
        
        # 保存划分信息
        split_info = {
            "split_mode": self.split_mode,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "modalities": modalities,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        info_file = self.output_root / 'split_info.json'
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"\n✓ 划分信息已保存: {info_file}")
        logger.info(f"\n{'='*70}")
        logger.info("转换完成！")
        logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='FRED 数据集转换为 COCO 格式 - YOLOv5 兼容版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 帧级别划分（推荐）
  python convert_fred_to_coco_v2.py --split-mode frame --modality both
  
  # 序列级别划分
  python convert_fred_to_coco_v2.py --split-mode sequence --modality both
  
  # 仅转换 RGB 模态
  python convert_fred_to_coco_v2.py --modality rgb
  
  # 简化模式（随机采样 10% 数据）
  python convert_fred_to_coco_v2.py --simple
  
  # 自定义划分比例
  python convert_fred_to_coco_v2.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """
    )
    
    parser.add_argument('--fred-root', type=str, 
                       default='/mnt/data/datasets/fred',
                       help='FRED 数据集根目录')
    parser.add_argument('--output-root', type=str, 
                       default='datasets/fred_coco',
                       help='输出目录')
    parser.add_argument('--split-mode', type=str, 
                       default='frame',
                       choices=['frame', 'sequence'],
                       help='划分模式: frame（帧级别）或 sequence（序列级别）')
    parser.add_argument('--modality', type=str, 
                       default='both',
                       choices=['rgb', 'event', 'both'],
                       help='转换的模态')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--simple', action='store_true',
                       help='启用简化模式，随机采样 10% 数据')
    parser.add_argument('--simple-ratio', type=float, default=0.1,
                       help='简化模式下的数据采样比例（默认 0.1）')
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"比例之和必须为 1.0，当前为 {total_ratio}")
        return 1
    
    try:
        converter = FREDtoCOCOConverter(
            fred_root=args.fred_root,
            output_root=args.output_root,
            split_mode=args.split_mode,
            simple=args.simple,
            simple_ratio=args.simple_ratio
        )
        
        converter.generate_all_splits(
            modality=args.modality,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        logger.info("\n✅ 所有转换完成！")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
