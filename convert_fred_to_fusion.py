#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED 数据集多模态融合转换脚本 - RGB + Event 配对 (v2)

主要改进：
1. 修复帧级别划分的实现
2. 添加帧级别划分的验证和统计
3. 改进时间戳配对的准确性
4. 添加数据划分的一致性检查

配对策略：
- RGB 帧作为基准，查找最近的 Event 帧
- 容差阈值：33ms（0.033秒）
- 配对状态：
  * DUAL：成功配对（时间差 ≤ 33ms）
  * RGB_ONLY：只找到 RGB 帧（无匹配的 Event）
  * EVENT_ONLY：只找到 Event 帧（无匹配的 RGB）

新的帧级别划分策略：
- 使用确定性哈希函数
- 根据序列ID和时间戳进行划分
- 确保同一帧总是分配到相同的数据集
- 可重现性：相同输入总是得到相同划分

使用方法：
    # 帧级别划分（已验证）
    python convert_fred_to_fusion_v2.py --split-mode frame
    
    # 序列级别划分
    python convert_fred_to_fusion_v2.py --split-mode sequence
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import random
import hashlib
from tqdm import tqdm
import logging
import shutil
from typing import List, Dict, Tuple, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameSplitValidator:
    """帧级别划分验证器"""
    
    def __init__(self, seed=42):
        self.seed = seed
        self.split_stats = {}
        self.frame_assignments = {}
        self.consistency_issues = []
    
    def get_frame_split(self, seq_id, frame_info, train_ratio=0.7, val_ratio=0.15):
        """
        确定帧的数据集划分
        
        Args:
            seq_id: 序列 ID
            frame_info: 帧信息字典
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            str: 'train', 'val', 或 'test'
        """
        # 创建唯一的帧标识符
        if frame_info.get('rgb_timestamp') is not None:
            # 使用 RGB 时间戳（主模态）
            timestamp = frame_info['rgb_timestamp']
            frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
        elif frame_info.get('event_timestamp') is not None:
            # 只有 Event 时间戳
            timestamp = frame_info['event_timestamp']
            frame_key = f"event_{seq_id}_{timestamp:.6f}"
        else:
            # 无时间戳，使用备用方案
            frame_key = f"unknown_{seq_id}_{random.randint(0, 1000000)}"
            logger.warning(f"帧无时间戳标识: {frame_key}")
        
        # 使用确定性哈希
        hash_input = f"{frame_key}_{self.seed}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        
        # 根据哈希值划分（确保一致性）
        mod_value = hash_value % 10000  # 使用更大的基数提高精度
        rand_val = mod_value / 10000.0
        
        if rand_val < train_ratio:
            return 'train', rand_val
        elif rand_val < train_ratio + val_ratio:
            return 'val', rand_val
        else:
            return 'test', rand_val
    
    def validate_frame_split(self, sequences, rgb_only_ratio=0.15, 
                           seed=42, num_checks=100):
        """
        验证帧级别划分的一致性
        
        Args:
            sequences: 序列 ID 列表
            rgb_only_ratio: 仅 RGB 帧的期望比例
            seed: 随机种子
            num_checks: 一致性检查次数
            
        Returns:
            dict: 验证结果
        """
        logger.info(f"\n{'='*70}")
        logger.info("帧级别划分验证")
        logger.info(f"{'='*70}")
        
        validation_results = {
            'seed_used': seed,
            'total_sequences': len(sequences),
            'consistency_checks_passed': 0,
            'consistency_checks_failed': 0,
            'assignment_stats': {},
            'errors': []
        }
        
        # 测试一致性：多次调用应得到相同结果
        consistency_passed = 0
        consistency_failed = 0
        
        for i in range(num_checks):
            # 随机选择一个序列和帧
            seq_id = random.choice(sequences) if sequences else 1
            
            # 模拟帧信息
            timestamp = random.uniform(0, 1000)
            frame_info = {
                'rgb_timestamp': timestamp,
                'event_timestamp': timestamp + random.uniform(-0.01, 0.01)
            }
            
            # 多次调用划分函数
            splits = []
            rand_vals = []
            for _ in range(3):
                split, rand_val = self.get_frame_split(seq_id, frame_info)
                splits.append(split)
                rand_vals.append(rand_val)
            
            # 检查是否一致
            if len(set(splits)) == 1 and len(set(rand_vals)) == 1:
                consistency_passed += 1
            else:
                consistency_failed += 1
                validation_results['errors'].append(
                    f"一致性失败: seq={seq_id}, splits={splits}, rand_vals={rand_vals}"
                )
        
        validation_results['consistency_checks_passed'] = consistency_passed
        validation_results['consistency_checks_failed'] = consistency_failed
        
        logger.info(f"\n一致性检查:")
        logger.info(f"  测试次数: {num_checks}")
        logger.info(f"  通过: {consistency_passed}")
        logger.info(f"  失败: {consistency_failed}")
        logger.info(f"  准确率: {consistency_passed/num_checks*100:.2f}%")
        
        if consistency_failed > 0:
            logger.warning("⚠️  一致性检查有失败案例！")
            validation_results['status'] = 'FAILED'
        else:
            logger.info("✅ 所有一致性检查通过！")
            validation_results['status'] = 'PASSED'
        
        return validation_results
    
    def analyze_frame_distribution(self, paired_frames, sequences, 
                                  train_ratio=0.7, val_ratio=0.15):
        """
        分析帧的数据集分布
        
        Args:
            paired_frames: 配对帧列表（按序列组织）
            sequences: 序列 ID 列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            dict: 分布统计
        """
        logger.info(f"\n{'='*70}")
        logger.info("帧级别划分分布分析")
        logger.info(f"{'='*70}")
        
        distribution = {
            'train': {'count': 0, 'dual': 0, 'rgb_only': 0, 'event_only': 0},
            'val': {'count': 0, 'dual': 0, 'rgb_only': 0, 'event_only': 0},
            'test': {'count': 0, 'dual': 0, 'rgb_only': 0, 'event_only': 0},
            'status_distribution': Counter(),
            'by_sequence': {}
        }
        
        # 按序列统计
        for seq_id in sequences:
            distribution['by_sequence'][seq_id] = {
                'total': 0,
                'train': 0,
                'val': 0,
                'test': 0
            }
        
        # 分析每个帧的划分
        for seq_id, frames in paired_frames.items():
            for frame_info in frames:
                split = self.get_frame_split(seq_id, frame_info)[0]
                status = frame_info.get('status', 'unknown')
                
                # 更新全局统计
                distribution[split]['count'] += 1
                distribution[split][status] = distribution[split].get(status, 0) + 1
                distribution['status_distribution'][split] += 1
                
                # 更新序列统计
                if seq_id in distribution['by_sequence']:
                    distribution['by_sequence'][seq_id][split] += 1
                    distribution['by_sequence'][seq_id]['total'] += 1
        
        # 打印分布
        total_frames = sum(dist['count'] for dist in distribution.values() 
                          if isinstance(dist, dict) and 'count' in dist)
        
        if total_frames > 0:
            logger.info(f"\n总体分布:")
            logger.info(f"  帧总数: {total_frames}")
            logger.info(f"  训练集: {distribution['train']['count']} "
                       f"({distribution['train']['count']/total_frames*100:.1f}%)")
            logger.info(f"  验证集: {distribution['val']['count']} "
                       f"({distribution['val']['count']/total_frames*100:.1f}%)")
            logger.info(f"  测试集: {distribution['test']['count']} "
                       f"({distribution['test']['count']/total_frames*100:.1f}%)")
            
            # 配对状态分布
            logger.info(f"\n配对状态分布:")
            for split in ['train', 'val', 'test']:
                if distribution[split]['count'] > 0:
                    dual_ratio = (distribution[split]['dual'] / 
                                 distribution[split]['count'] * 100)
                    rgb_ratio = (distribution[split]['rgb_only'] / 
                                distribution[split]['count'] * 100)
                    event_ratio = (distribution[split]['event_only'] / 
                                  distribution[split]['count'] * 100)
                    
                    logger.info(f"  {split.upper()}:")
                    logger.info(f"    双模态: {distribution[split]['dual']} ({dual_ratio:.1f}%)")
                    logger.info(f"    仅 RGB: {distribution[split]['rgb_only']} ({rgb_ratio:.1f}%)")
                    logger.info(f"    仅 Event: {distribution[split]['event_only']} ({event_ratio:.1f}%)")
        
        return distribution
    
    def validate_from_data(self, paired_frames_data, num_checks=100):
        """
        使用已有数据验证划分一致性（不重复处理序列）
        
        Args:
            paired_frames_data: 已处理的帧数据，格式为 {seq_id: [frame_info, ...]}
            num_checks: 验证次数
            
        Returns:
            dict: 验证结果
        """
        import random
        from collections import Counter
        
        logger.info(f"\n{'='*70}")
        logger.info("帧级别划分验证（使用已有数据）")
        logger.info(f"{'='*70}")
        
        if not paired_frames_data:
            logger.warning("⚠️  没有可用的帧数据，跳过验证")
            return {'status': 'SKIPPED', 'errors': ['No data provided']}
        
        seq_ids = list(paired_frames_data.keys())
        
        validation_results = {
            'seed_used': self.seed,
            'total_sequences': len(seq_ids),
            'consistency_checks_passed': 0,
            'consistency_checks_failed': 0,
            'assignment_stats': {},
            'errors': []
        }
        
        # 测试一致性：多次调用应得到相同结果
        consistency_passed = 0
        consistency_failed = 0
        
        for i in range(num_checks):
            # 随机选择一个序列和帧
            seq_id = random.choice(seq_ids)
            frames = paired_frames_data[seq_id]
            
            if not frames:
                continue
                
            # 随机选择一个帧
            frame_info = random.choice(frames)
            
            # 多次调用划分函数
            splits = []
            rand_vals = []
            for _ in range(3):
                split, rand_val = self.get_frame_split(seq_id, frame_info)
                splits.append(split)
                rand_vals.append(rand_val)
            
            # 检查是否一致
            if len(set(splits)) == 1 and len(set(rand_vals)) == 1:
                consistency_passed += 1
            else:
                consistency_failed += 1
                validation_results['errors'].append(
                    f"一致性失败: seq={seq_id}, splits={splits}, rand_vals={rand_vals}"
                )
        
        validation_results['consistency_checks_passed'] = consistency_passed
        validation_results['consistency_checks_failed'] = consistency_failed
        
        logger.info(f"\n一致性检查:")
        logger.info(f"  测试次数: {num_checks}")
        logger.info(f"  通过: {consistency_passed}")
        logger.info(f"  失败: {consistency_failed}")
        if num_checks > 0:
            logger.info(f"  准确率: {consistency_passed/num_checks*100:.2f}%")
        
        if consistency_failed > 0:
            logger.warning("⚠️  一致性检查有失败案例！")
            validation_results['status'] = 'FAILED'
        else:
            logger.info("✅ 所有一致性检查通过！")
            validation_results['status'] = 'PASSED'
        
        return validation_results


class FREDFusionConverterV2:
    """FRED 数据集多模态融合转换器 v2"""
    
    def __init__(self, fred_root, output_root, split_mode='frame', 
                 threshold=0.033, simple=False, simple_ratio=0.1,
                 seed=42, use_symlinks=False):
        """
        初始化融合转换器
        
        Args:
            fred_root: FRED 数据集根目录
            output_root: 输出目录
            split_mode: 'frame' 或 'sequence'
            threshold: 时间戳配对容差阈值（秒，默认 33ms）
            simple: 是否启用简化模式
            simple_ratio: 简化模式下的采样比例
            use_symlinks: 是否使用软链接而非拷贝图片
        """
        self.fred_root = Path(fred_root)
        self.output_root = Path(output_root)
        self.split_mode = split_mode
        self.threshold = threshold  # 秒
        self.simple = simple
        self.simple_ratio = simple_ratio
        self.seed = seed
        self.use_symlinks = use_symlinks
        
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
        
        # 帧划分验证器
        self.validator = FrameSplitValidator(seed=seed)
        
        logger.info(f"FRED 根目录: {self.fred_root}")
        logger.info(f"输出目录: {self.output_root}")
        logger.info(f"划分模式: {split_mode}")
        logger.info(f"时间配对容差: {self.threshold * 1000:.1f} ms")
    
    def get_all_sequences(self):
        """获取所有可用的序列 ID"""
        sequences = []
        for seq_dir in sorted(self.fred_root.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                sequences.append(int(seq_dir.name))
        return sorted(sequences)
    
    def _sample_frames(self, all_sequences_data):
        """
        按比例采样帧（用于简化模式）
        
        Args:
            all_sequences_data: 所有序列的帧数据，格式为 {seq_id: [frame_info, ...]}
            
        Returns:
            dict: 采样后的帧数据，格式为 {seq_id: [frame_info, ...]}
        """
        if self.simple_ratio >= 1.0:
            return all_sequences_data
        
        # 收集所有帧
        all_frames = []
        for seq_id, frames in all_sequences_data.items():
            for frame_info in frames:
                # 检查是否有标注
                timestamp = (frame_info['rgb_timestamp'] if frame_info['rgb_timestamp'] is not None 
                           else frame_info['event_timestamp'])
                all_frames.append((seq_id, frame_info, timestamp))
        
        # 计算采样数量
        n_sample = max(1, int(len(all_frames) * self.simple_ratio))
        
        # 随机采样（可重现）
        random.seed(42)
        sampled_indices = set(random.sample(range(len(all_frames)), n_sample))
        
        # 重建采样后的数据结构
        sampled_data = {}
        for idx, (seq_id, frame_info, timestamp) in enumerate(all_frames):
            if idx in sampled_indices:
                if seq_id not in sampled_data:
                    sampled_data[seq_id] = []
                sampled_data[seq_id].append(frame_info)
        
        logger.info(f"帧级别采样: 从 {len(all_frames)} 帧中采样 {n_sample} 帧")
        return sampled_data
    
    def load_annotations(self, sequence_path):
        """加载标注文件"""
        annotation_file = sequence_path / "interpolated_coordinates.txt"
        
        if not annotation_file.exists():
            annotation_file = sequence_path / "coordinates.txt"
        
        if not annotation_file.exists():
            logger.warning(f"序列 {sequence_path.name} 无标注文件")
            return {}
        
        annotations = {}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    time_str, coords_str = line.split(': ')
                    timestamp = float(time_str)
                    coords = [x.strip() for x in coords_str.split(',')]
                    x1, y1, x2, y2 = map(float, coords[:4])
                    drone_id = int(float(coords[4])) if len(coords) > 4 else 1
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    if timestamp not in annotations:
                        annotations[timestamp] = []
                    
                    annotations[timestamp].append({
                        'bbox': (x1, y1, x2, y2),
                        'drone_id': drone_id,
                        'category_id': 1,
                        'area': (x2 - x1) * (y2 - y1)
                    })
                    
                except Exception as e:
                    continue
        
        logger.info(f"序列 {sequence_path.name}: 加载 {len(annotations)} 个时间戳的标注")
        return annotations
    
    def get_frames(self, sequence_path, modality):
        """获取帧列表及其时间戳"""
        if modality == 'rgb':
            frame_dir = sequence_path / "PADDED_RGB"
            if not frame_dir.exists():
                frame_dir = sequence_path / "RGB"
            pattern = "*.jpg"
        elif modality == 'event':
            frame_dir = sequence_path / "Event" / "Frames"
            pattern = "*.png"
        else:
            raise ValueError(f"未知模态: {modality}")
        
        if not frame_dir.exists():
            return []
        
        frames = []
        
        # 对于 Event 模态，需要按帧号排序而不是文件名排序
        # 因为文件名是字母顺序（Video_0_frame_100... < Video_0_frame_333...）
        if modality == 'event':
            import re
            def _extract_frame_number(filename):
                match = re.search(r'frame_(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            frame_paths = sorted(frame_dir.glob(pattern), key=lambda x: _extract_frame_number(x.name))
        else:
            frame_paths = sorted(frame_dir.glob(pattern))
        
        for frame_path in frame_paths:
            timestamp = self._extract_timestamp(frame_path.name, modality)
            if timestamp is not None:
                frames.append((timestamp, frame_path))
        
        frames = sorted(frames, key=lambda x: x[0])
        
        # 加载标注文件以获取参考时间戳
        annotations_dict = self.load_annotations(sequence_path)
        
        # 对于 RGB 模态，将绝对时间戳转换为相对时间戳
        if modality == 'rgb' and frames and annotations_dict:
            # 获取标注中的最小时间戳
            first_ann_ts = min(annotations_dict.keys())
            
            # 从 RGB 帧的时间戳计算偏移量
            first_rgb_abs_ts = frames[0][0]
            offset = first_rgb_abs_ts - first_ann_ts
            
            # 转换所有 RGB 帧为相对时间戳
            frames = [(rgb_abs_ts - offset, path) for rgb_abs_ts, path in frames]
        
        # 对于 Event 模态，调整时间戳以与 RGB/标注对齐
        elif modality == 'event' and frames and annotations_dict:
            # 获取标注中的最小时间戳
            first_ann_ts = min(annotations_dict.keys())
            
            # 计算偏移量：第一个 RGB 帧的时间 - 第一个 Event 帧的时间
            # 先获取 RGB 帧的时间戳
            rgb_frames = []
            rgb_dir = sequence_path / "PADDED_RGB"
            if not rgb_dir.exists():
                rgb_dir = sequence_path / "RGB"
            
            for f in sorted(rgb_dir.glob("*.jpg"))[:1]:  # 只需要第一个 RGB 帧
                rgb_ts = self._extract_timestamp(f.name, 'rgb')
                if rgb_ts is not None:
                    # 计算 RGB 帧的偏移量（与上面 RGB 的处理相同）
                    offset_rgb = rgb_ts - first_ann_ts
                    first_rgb_rel_ts = rgb_ts - offset_rgb
                    event_start_timestamp = first_rgb_rel_ts
                    break
            
            if 'event_start_timestamp' in locals():
                first_event_abs_ts = frames[0][0]
                event_offset = event_start_timestamp - first_event_abs_ts
                
                # 转换所有 Event 帧为相对时间戳
                frames = [(event_abs_ts + event_offset, path) for event_abs_ts, path in frames]
        
        return frames
    
    def _extract_timestamp(self, filename, modality):
        """从文件名提取时间戳（返回绝对时间戳，单位：秒）"""
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
                    # 返回绝对时间戳（包含小时信息）
                    return hours * 3600 + minutes * 60 + seconds
            
            elif modality == 'event':
                # Video_0_frame_100032333.png
                name = filename.replace('.png', '')
                parts = name.split('_')
                if len(parts) >= 3:
                    timestamp_us = int(parts[-1])
                    # 转换为秒（绝对时间戳，从视频开始计算）
                    return timestamp_us / 1_000_000
        except Exception as e:
            pass
        return None
    
    def pair_timestamps(self, rgb_timestamps: List[Tuple[float, Path]], 
                       event_timestamps: List[Tuple[float, Path]]) -> List[Dict]:
        """根据时间戳配对 RGB 和 Event 帧"""
        paired = []
        used_event_indices = set()
        
        # 以 RGB 为基准进行配对
        pbar = tqdm(rgb_timestamps, desc="配对 RGB-EVENT")
        for rgb_idx, (rgb_ts, rgb_path) in enumerate(pbar):
            best_match = None
            min_diff = float('inf')
            best_event_idx = -1
            
            # 查找最接近的 Event 帧
            for event_idx, (event_ts, event_path) in enumerate(event_timestamps):
                if event_idx in used_event_indices:
                    continue
                
                time_diff = abs(rgb_ts - event_ts)
                if time_diff < min_diff:
                    min_diff = time_diff
                    best_match = (event_ts, event_path)
                    best_event_idx = event_idx
            
            # 判断是否满足容差阈值
            if best_match and min_diff <= self.threshold:
                paired.append({
                    'rgb_timestamp': rgb_ts,
                    'rgb_path': rgb_path,
                    'event_timestamp': best_match[0],
                    'event_path': best_match[1],
                    'time_diff': min_diff,
                    'status': 'dual'
                })
                used_event_indices.add(best_event_idx)
                pbar.set_postfix({'dual': len([p for p in paired if p['status'] == 'dual'])})
            else:
                # 无匹配的 Event 帧，计算与最近 Event 帧的时间差
                actual_diff = min_diff if best_match else float('inf')
                if actual_diff == float('inf') and event_timestamps:  # 如果有 Event 帧但没有匹配
                    actual_diff = min(abs(rgb_ts - event_ts) for event_ts, _ in event_timestamps)
                
                paired.append({
                    'rgb_timestamp': rgb_ts,
                    'rgb_path': rgb_path,
                    'event_timestamp': None,
                    'event_path': None,
                    'time_diff': actual_diff,
                    'status': 'rgb_only'
                })
                pbar.set_postfix({
                    'dual': len([p for p in paired if p['status'] == 'dual']),
                    'rgb_only': len([p for p in paired if p['status'] == 'rgb_only'])
                })
        
        # 添加未匹配的 Event 帧
        for event_idx, (event_ts, event_path) in enumerate(event_timestamps):
            if event_idx not in used_event_indices:
                # 计算与最近的 RGB 帧的时间差
                min_diff = float('inf')
                if rgb_timestamps:  # 如果有 RGB 帧
                    for rgb_ts, _ in rgb_timestamps:
                        diff = abs(rgb_ts - event_ts)
                        if diff < min_diff:
                            min_diff = diff
                
                paired.append({
                    'rgb_timestamp': None,
                    'rgb_path': None,
                    'event_timestamp': event_ts,
                    'event_path': event_path,
                    'time_diff': min_diff,
                    'status': 'event_only'
                })
        
        return paired
    
    def process_sequence(self, sequence_id):
        """处理单个序列"""
        sequence_path = self.fred_root / str(sequence_id)
        
        if not sequence_path.exists():
            return {}, [], {}
        
        # 加载标注
        annotations_dict = self.load_annotations(sequence_path)
        
        # 获取 RGB 和 Event 帧
        rgb_frames = self.get_frames(sequence_path, 'rgb')
        event_frames = self.get_frames(sequence_path, 'event')
        
        if not rgb_frames and not event_frames:
            return {}, [], {}
        
        logger.info(f"序列 {sequence_id}: {len(rgb_frames)} RGB 帧, {len(event_frames)} Event 帧")
        
        # 配对时间戳
        paired_frames = self.pair_timestamps(rgb_frames, event_frames)
        
        # 为每个帧添加文件名信息
        for frame_info in paired_frames:
            # 添加文件名信息（兼容标准 COCO 格式）
            if frame_info.get('rgb_path') and frame_info.get('event_path'):
                # 双模态：使用 RGB 作为主 file_name（兼容性）
                frame_info['rgb_file_name'] = str(frame_info['rgb_path'].relative_to(self.fred_root))
                frame_info['event_file_name'] = str(frame_info['event_path'].relative_to(self.fred_root))
                frame_info['modality'] = 'dual'
            elif frame_info.get('rgb_path'):
                # 仅 RGB
                frame_info['rgb_file_name'] = str(frame_info['rgb_path'].relative_to(self.fred_root))
                frame_info['modality'] = 'rgb'
            elif frame_info.get('event_path'):
                # 仅 Event
                frame_info['event_file_name'] = str(frame_info['event_path'].relative_to(self.fred_root))
                frame_info['modality'] = 'event'
        
        return annotations_dict, paired_frames, None
    
    def split_sequences(self, sequences, train_ratio=0.7, val_ratio=0.15, 
                       test_ratio=0.15, seed=42):
        """序列级别划分"""
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
    
    def generate_frame_level_split(self, sequences, train_ratio=0.7, val_ratio=0.15):
        """
        生成帧级别划分
        
        Args:
            sequences: 序列 ID 列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            dict: {train: [frames], val: [frames], test: [frames]}
        """
        logger.info(f"\n生成帧级别划分...")
        
        # 收集所有序列的帧和标注
        all_sequences_data = {}
        all_annotations = {}
        
        for seq_id in tqdm(sequences, desc="处理序列"):
            annotations_dict, paired_frames, images = self.process_sequence(seq_id)
            all_sequences_data[seq_id] = paired_frames
            all_annotations[seq_id] = annotations_dict
        
        # 简化模式：在所有帧中采样，而不是采样序列
        if self.simple:
            logger.info(f"\n应用帧级别简化采样（比例: {self.simple_ratio:.1%}）...")
            total_frames_before = sum(len(frames) for frames in all_sequences_data.values())
            all_sequences_data = self._sample_frames(all_sequences_data)
            total_frames_after = sum(len(frames) for frames in all_sequences_data.values())
            logger.info(f"采样结果: {total_frames_before} -> {total_frames_after} 帧")
        
        # 验证划分（跳过重复处理）
        logger.info("\n验证帧级别划分一致性...")
        # 使用已处理的数据进行验证，避免重复调用 process_sequence
        validation_result = self.validator.validate_from_data(all_sequences_data)
        
        if validation_result['status'] != 'PASSED':
            logger.warning("⚠️  帧级别划分验证未通过，但将继续生成数据集")
        
        # 分析分布
        distribution = self.validator.analyze_frame_distribution(
            all_sequences_data, sequences, train_ratio, val_ratio
        )
        
        # 划分帧
        split_results = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for seq_id, frames in all_sequences_data.items():
            annotations_dict = all_annotations[seq_id]  # 使用已加载的标注
            
            for frame_info in frames:
                # 获取时间戳
                timestamp = (frame_info['rgb_timestamp'] if frame_info['rgb_timestamp'] is not None 
                           else frame_info['event_timestamp'])
                
                # 如果没有时间戳（异常情况），跳过
                if timestamp is None:
                    continue
                
                # 查找匹配的标注（使用容差匹配处理浮点精度问题）
                match_timestamp = None
                min_diff = float('inf')
                
                # 尝试在标注字典的所有键中查找匹配
                for ann_ts in annotations_dict.keys():
                    # ann_ts 可能是字符串或浮点数，需要类型转换比较
                    try:
                        ann_ts_float = float(ann_ts)
                    except (ValueError, TypeError):
                        continue
                    
                    diff = abs(ann_ts_float - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        match_timestamp = ann_ts
                    if min_diff < 0.001:  # 1ms 容差
                        break
                
                # 如果没有找到匹配的标注（最小差值大于 10ms），跳过
                if min_diff > 0.01:
                    continue
                
                # 确定划分
                split = self.validator.get_frame_split(seq_id, frame_info, train_ratio, val_ratio)[0]
                split_results[split].append((seq_id, frame_info))
        
        logger.info(f"\n帧级别划分结果:")
        for split_name, frames in split_results.items():
            logger.info(f"  {split_name.upper()}: {len(frames)} 帧")
        
        return split_results, all_annotations
    
    def generate_sequence_level_split(self, sequences, train_ratio=0.7, val_ratio=0.15, 
                                    test_ratio=0.15, seed=42):
        """生成序列级别划分"""
        logger.info(f"\n生成序列级别划分...")
        
        train_seqs, val_seqs, test_seqs = self.split_sequences(
            sequences, train_ratio, val_ratio, test_ratio, seed
        )
        
        logger.info(f"\n序列划分:")
        logger.info(f"  训练: {len(train_seqs)} 序列")
        logger.info(f"  验证: {len(val_seqs)} 序列")
        logger.info(f"  测试: {len(test_seqs)} 序列")
        
        return {
            'train': [(seq_id, None) for seq_id in train_seqs],
            'val': [(seq_id, None) for seq_id in val_seqs],
            'test': [(seq_id, None) for seq_id in test_seqs]
        }
    
    def generate_annotation_records(self, split_frames, all_annotations=None):
        """
        生成 COCO 格式标注记录
        
        Args:
            split_frames: 划分后的帧列表
            
        Returns:
            tuple: (image_records, annotation_records)
        """
        images = []
        annotations = []
        
        image_id = 1
        annotation_id = 1
        
        for seq_id, frame_info in split_frames:
            sequence_path = self.fred_root / str(seq_id)
            
            # 加载标注（使用缓存或重新加载）
            if all_annotations and seq_id in all_annotations:
                annotations_dict = all_annotations[seq_id]
            else:
                annotations_dict, _, _ = self.process_sequence(seq_id)
            
            # 获取时间戳
            timestamp = (frame_info['rgb_timestamp'] if frame_info['rgb_timestamp'] is not None 
                        else frame_info['event_timestamp'])
            
            # 查找匹配的标注（处理浮点数匹配问题）
            anns = []
            min_diff = float('inf')
            match_ann_timestamp = None
            
            for ann_ts in annotations_dict.keys():
                try:
                    ann_ts_float = float(ann_ts)
                except (ValueError, TypeError):
                    continue
                
                diff = abs(ann_ts_float - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    match_ann_timestamp = ann_ts
                if min_diff < 0.001:  # 1ms 容差
                    break
            
            # 如果找到匹配，获取标注
            if match_ann_timestamp is not None and min_diff < 0.01:  # 10ms 容差
                anns = annotations_dict.get(match_ann_timestamp, [])
            
            if not anns:
                continue
            
            # 创建图像记录
            image_entry = {
                "id": image_id,
                "width": 1280,
                "height": 720,
                "sequence_id": seq_id,
                "rgb_timestamp": frame_info['rgb_timestamp'],
                "event_timestamp": frame_info['event_timestamp'],
                "time_diff": frame_info['time_diff'],
                "status": frame_info['status'],
                "modality": frame_info.get('modality', 'unknown')
            }
            
            # 添加文件名信息（兼容标准 COCO 格式）
            if frame_info.get('rgb_file_name') and frame_info.get('event_file_name'):
                # 双模态：使用 RGB 作为主 file_name
                image_entry['file_name'] = frame_info['rgb_file_name']
                image_entry['rgb_file_name'] = frame_info['rgb_file_name']
                image_entry['event_file_name'] = frame_info['event_file_name']
            elif frame_info.get('rgb_file_name'):
                # 仅 RGB
                image_entry['file_name'] = frame_info['rgb_file_name']
                image_entry['rgb_file_name'] = frame_info['rgb_file_name']
            elif frame_info.get('event_file_name'):
                # 仅 Event
                image_entry['file_name'] = frame_info['event_file_name']
                image_entry['event_file_name'] = frame_info['event_file_name']
            else:
                # 没有文件名信息，从路径重新生成
                if frame_info.get('rgb_path') and frame_info.get('event_path'):
                    image_entry['rgb_file_name'] = str(frame_info['rgb_path'].relative_to(self.fred_root))
                    image_entry['event_file_name'] = str(frame_info['event_path'].relative_to(self.fred_root))
                    image_entry['file_name'] = image_entry['rgb_file_name']
                elif frame_info.get('rgb_path'):
                    image_entry['rgb_file_name'] = str(frame_info['rgb_path'].relative_to(self.fred_root))
                    image_entry['file_name'] = image_entry['rgb_file_name']
                elif frame_info.get('event_path'):
                    image_entry['event_file_name'] = str(frame_info['event_path'].relative_to(self.fred_root))
                    image_entry['file_name'] = image_entry['event_file_name']
            
            images.append(image_entry)
            
            # 创建标注记录
            for ann in anns:
                # 验证边界框
                x1, y1, x2, y2 = ann['bbox']
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 转换为 COCO 格式
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann['category_id'],
                    "bbox": bbox_coco,
                    "area": area,
                    "iscrowd": 0,
                    "drone_id": ann['drone_id'],
                    "modality": image_entry["modality"]
                }
                annotations.append(annotation_entry)
                annotation_id += 1
            
            image_id += 1
        
        return images, annotations
    
    def generate_all_fusion(self, train_ratio=0.7, val_ratio=0.15, 
                           test_ratio=0.15, seed=42, simple_mode=False):
        """生成完整融合数据集"""
        logger.info(f"\n{'='*70}")
        logger.info(f"FRED 融合转换 - {self.split_mode.upper()} 级别划分")
        logger.info(f"{'='*70}")
        
        if self.simple:
            logger.info(f"⚠️  简化模式已启用！采样比例: {self.simple_ratio:.1%}")
            logger.info(f"  This will use only {self.simple_ratio:.1%} of the data for quick testing")
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "annotations").mkdir(exist_ok=True)
        
        # 获取所有序列
        sequences = self.get_all_sequences()
        logger.info(f"找到 {len(sequences)} 个序列: {sequences}")
        
        # 注意：简化模式现在在帧级别进行采样，而不是序列级别
        # 详见 generate_frame_level_split 方法中的实现
        
        # 生成划分
        if self.split_mode == 'frame':
            split_results, all_annotations = self.generate_frame_level_split(sequences, train_ratio, val_ratio)
        else:
            split_results = self.generate_sequence_level_split(
                sequences, train_ratio, val_ratio, test_ratio, seed
            )
            all_annotations = None  # 序列级别不需要预先加载标注
        
        # 生成标注文件
        for split_name, split_frames in split_results.items():
            if not split_frames:
                logger.warning(f"{split_name.upper()} 划分无有效帧，跳过")
                continue
            
            logger.info(f"\n处理 {split_name.upper()} 划分...")
            
            # 生成 COCO 格式记录
            if self.split_mode == 'frame' and all_annotations is not None:
                images, annotations = self.generate_annotation_records(split_frames, all_annotations)
            else:
                images, annotations = self.generate_annotation_records(split_frames)
            
            # 创建 COCO 字典
            coco_dict = {
                "info": {
                    "description": f"FRED Dataset - Fusion (RGB + Event) - {split_name.upper()} split ({self.split_mode}-level)",
                    "version": "2.0",
                    "year": 2025,
                    "contributor": "MICC - University of Florence",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fusion_threshold_ms": int(self.threshold * 1000),
                    "split_mode": self.split_mode
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Apache License 2.0",
                        "url": "https://www.apache.org/licenses/LICENSE-2.0"
                    }
                ],
                "categories": self.categories,
                "images": images,
                "annotations": annotations,
                "fusion_modes": {
                    'dual': 1,
                    'rgb': 2,
                    'event': 3,
                    'both_failed': 4
                }
            }
            
            # 保存文件
            output_file = self.output_root / "annotations" / f'instances_{split_name}.json'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ 已保存: {output_file}")
            logger.info(f"  图像: {len(images)}, 标注: {len(annotations)}")
        
        # 保存融合信息
        fusion_info = {
            "dataset_type": "fusion_v2",
            "split_mode": self.split_mode,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "fusion_threshold_ms": int(self.threshold * 1000),
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "validator_status": str(self.validator.validate_frame_split(sequences))[:100]  # 只保存前100个字符
        }
        
        info_file = self.output_root / 'fusion_info_v2.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(fusion_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ 融合信息已保存: {info_file}")
        logger.info(f"\n{'='*70}")
        logger.info("融合数据集生成完成！")
        logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='FRED 数据集多模态融合转换 v2 - 已验证的帧级别划分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 帧级别划分（已验证）
  python convert_fred_to_fusion_v2.py --split-mode frame
  
  # 简化模式（快速测试，采样 10% 数据）
  python convert_fred_to_fusion_v2.py --split-mode frame --simple
  
  # 自定义采样比例（5% 数据）
  python convert_fred_to_fusion_v2.py --split-mode frame --simple --simple-ratio 0.05
  
  # 序列级别划分
  python convert_fred_to_fusion_v2.py --split-mode sequence
  
  # 自定义时间容差
  python convert_fred_to_fusion_v2.py --threshold 0.02
  
  # 验证帧级别划分
  python convert_fred_to_fusion_v2.py --split-mode frame --validate-only
        """
    )
    
    parser.add_argument('--fred-root', type=str, 
                       default='/mnt/data/datasets/fred',
                       help='FRED 数据集根目录')
    parser.add_argument('--output-root', type=str, 
                       default='datasets/fred_fusion',
                       help='输出目录')
    parser.add_argument('--split-mode', type=str, 
                       default='frame',
                       choices=['frame', 'sequence'],
                       help='划分模式（frame 或 sequence）')
    parser.add_argument('--threshold', type=float, 
                       default=0.033,
                       help='时间配对容差阈值（秒，默认 33ms）')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅验证帧级别划分（不生成数据集）')
    parser.add_argument('--simple', action='store_true',
                       help='启用简化模式（快速测试，采样部分数据）')
    parser.add_argument('--simple-ratio', type=float, default=0.1,
                       help='简化模式下的采样比例（默认 0.1 = 10%）')
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"比例之和必须为 1.0，当前为 {total_ratio}")
        return 1
    
    try:
        converter = FREDFusionConverterV2(
            fred_root=args.fred_root,
            output_root=args.output_root,
            split_mode=args.split_mode,
            threshold=args.threshold,
            simple=args.simple,
            simple_ratio=args.simple_ratio,
            seed=args.seed
        )
        
        if args.validate_only:
            # 仅验证
            sequences = converter.get_all_sequences()
            validator = FrameSplitValidator(seed=args.seed)
            result = validator.validate_frame_split(sequences)
            logger.info(f"\n验证结果: {result['status']}")
            return 0 if result['status'] == 'PASSED' else 1
        
        # 生成数据集
        converter.generate_all_fusion(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        logger.info("\n✅ 多模态融合数据集 v2 生成完成！")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())