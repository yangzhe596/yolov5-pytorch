#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED 数据集序列可视化工具

功能：
1. 可视化整个视频序列的标注
2. 支持导出为视频文件
3. 支持 RGB 和 Event 两种模态
4. 显示边界框、drone_id、时间戳等信息
5. 支持选择特定序列或随机序列

使用方法：
    # 可视化单个序列并导出视频
    python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video
    
    # 可视化随机序列（不导出视频）
    python visualize_fred_sequences.py --modality event --random
    
    # 可视化多个序列
    python visualize_fred_sequences.py --modality rgb --sequences 0 1 5 --export-video
    
    # 自定义输出目录和帧率
    python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video --output-dir visualizations --fps 30
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FREDSequenceVisualizer:
    """FRED 序列可视化器"""
    
    def __init__(self, fred_root, coco_root, modality='rgb'):
        """
        初始化可视化器
        
        Args:
            fred_root: FRED 数据集根目录
            coco_root: COCO 格式数据集根目录
            modality: 'rgb' 或 'event'
        """
        self.fred_root = Path(fred_root)
        self.coco_root = Path(coco_root)
        self.modality = modality.lower()
        
        if not self.fred_root.exists():
            raise FileNotFoundError(f"FRED 根目录不存在: {self.fred_root}")
        
        if not self.coco_root.exists():
            raise FileNotFoundError(f"COCO 根目录不存在: {self.coco_root}")
        
        # 加载 COCO 标注
        self.annotations = self._load_all_annotations()
        
        # 颜色配置（BGR 格式）
        self.colors = {
            1: (0, 255, 0),    # drone_id 1: 绿色
            2: (255, 0, 0),    # drone_id 2: 蓝色
            3: (0, 0, 255),    # drone_id 3: 红色
            4: (255, 255, 0),  # drone_id 4: 青色
            5: (255, 0, 255),  # drone_id 5: 品红色
        }
        
        logger.info(f"初始化完成 - 模态: {self.modality}")
        logger.info(f"FRED 根目录: {self.fred_root}")
        logger.info(f"COCO 根目录: {self.coco_root}")
    
    def _load_all_annotations(self):
        """加载所有划分的 COCO 标注（优化版本）"""
        logger.info("正在加载 COCO 标注...")
        annotations = {}
        
        # 第一步：创建全局 image_id 到 sequence_id 的映射
        image_id_to_seq = {}
        
        for split in ['train', 'val', 'test']:
            json_file = self.coco_root / self.modality / 'annotations' / f'instances_{split}.json'
            
            if not json_file.exists():
                logger.warning(f"标注文件不存在: {json_file}")
                continue
            
            logger.info(f"  加载 {split} 划分...")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"    图像数: {len(data['images'])}, 标注数: {len(data['annotations'])}")
            
            # 按序列组织图像数据
            for img in data['images']:
                seq_id = img['sequence_id']
                if seq_id not in annotations:
                    annotations[seq_id] = {
                        'images': [],
                        'annotations': defaultdict(list)
                    }
                annotations[seq_id]['images'].append(img)
                image_id_to_seq[img['id']] = seq_id
            
            # 组织标注（优化：使用全局映射表，O(1) 查找）
            for ann in data['annotations']:
                img_id = ann['image_id']
                seq_id = image_id_to_seq.get(img_id)
                if seq_id is not None:
                    annotations[seq_id]['annotations'][img_id].append(ann)
        
        # 按时间戳排序每个序列的图像
        logger.info("正在排序序列...")
        for seq_id in annotations:
            annotations[seq_id]['images'].sort(key=lambda x: x['timestamp'])
        
        logger.info(f"✓ 加载完成！共 {len(annotations)} 个序列")
        return annotations
    
    def get_available_sequences(self):
        """获取所有可用的序列 ID"""
        return sorted(self.annotations.keys())
    
    def get_sequence_info(self, sequence_id):
        """获取序列信息"""
        if sequence_id not in self.annotations:
            return None
        
        seq_data = self.annotations[sequence_id]
        n_images = len(seq_data['images'])
        n_annotations = sum(len(anns) for anns in seq_data['annotations'].values())
        
        # 获取时间范围
        timestamps = [img['timestamp'] for img in seq_data['images']]
        duration = max(timestamps) - min(timestamps) if timestamps else 0
        
        # 获取 drone_ids
        drone_ids = set()
        for anns in seq_data['annotations'].values():
            for ann in anns:
                drone_ids.add(ann.get('drone_id', 1))
        
        return {
            'sequence_id': sequence_id,
            'n_images': n_images,
            'n_annotations': n_annotations,
            'duration': duration,
            'drone_ids': sorted(drone_ids),
            'avg_annotations_per_image': n_annotations / n_images if n_images > 0 else 0
        }
    
    def visualize_sequence(self, sequence_id, export_video=False, output_dir='visualizations',
                          fps=30, show_window=True, max_frames=None):
        """
        可视化序列
        
        Args:
            sequence_id: 序列 ID
            export_video: 是否导出为视频
            output_dir: 输出目录
            fps: 视频帧率
            show_window: 是否显示窗口
            max_frames: 最大帧数（用于快速预览）
        
        Returns:
            dict: 可视化统计信息
        """
        if sequence_id not in self.annotations:
            logger.error(f"序列 {sequence_id} 不存在")
            return None
        
        seq_data = self.annotations[sequence_id]
        images = seq_data['images']
        
        original_count = len(images)
        if max_frames:
            images = images[:max_frames]
            logger.info(f"⚠️  快速预览模式：仅处理前 {max_frames} 帧（共 {original_count} 帧）")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎬 可视化序列 {sequence_id} ({self.modality.upper()} 模态)")
        logger.info(f"{'='*70}")
        logger.info(f"📊 总帧数: {len(images)}")
        logger.info(f"📹 导出视频: {'是' if export_video else '否'}")
        logger.info(f"🖥️  显示窗口: {'是' if show_window else '否'}")
        if export_video:
            logger.info(f"📁 输出目录: {output_dir}")
            logger.info(f"🎞️  帧率: {fps} FPS")
        
        # 创建输出目录
        if export_video:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ 输出目录已创建: {output_path}")
            
            # 视频输出路径
            video_file = output_path / f"sequence_{sequence_id}_{self.modality}.mp4"
            
            # 获取第一帧以确定视频尺寸
            logger.info("正在读取第一帧以确定视频尺寸...")
            first_img_path = self.fred_root / images[0]['file_name']
            first_frame = cv2.imread(str(first_img_path))
            if first_frame is None:
                logger.error(f"❌ 无法读取第一帧: {first_img_path}")
                return None
            
            height, width = first_frame.shape[:2]
            logger.info(f"✓ 视频分辨率: {width}x{height}")
            
            # 创建视频写入器
            logger.info("正在初始化视频编码器...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                logger.error("❌ 无法创建视频写入器")
                return None
            
            logger.info(f"✓ 视频写入器已就绪")
            logger.info(f"📹 输出文件: {video_file}")
        else:
            video_writer = None
        
        # 统计信息
        stats = {
            'sequence_id': sequence_id,
            'total_frames': len(images),
            'frames_with_annotations': 0,
            'total_annotations': 0,
            'drone_ids': set()
        }
        
        # 处理每一帧
        logger.info(f"开始处理 {len(images)} 帧...")
        
        # 创建进度条
        pbar = tqdm(images, desc=f"序列 {sequence_id}", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                   ncols=100)
        
        processed_frames = 0
        skipped_frames = 0
        
        for idx, img_info in enumerate(pbar):
            # 读取图像
            img_path = self.fred_root / img_info['file_name']
            frame = cv2.imread(str(img_path))
            
            if frame is None:
                skipped_frames += 1
                if skipped_frames <= 5:  # 只显示前5个错误
                    logger.warning(f"无法读取图像: {img_path}")
                continue
            
            processed_frames += 1
            
            # 更新进度条描述（减少更新频率以提升性能）
            if idx % 100 == 0:
                pbar.set_postfix({
                    '已处理': processed_frames,
                    '跳过': skipped_frames,
                    '时间': f"{img_info['timestamp']:.1f}s"
                })
            
            # 获取该帧的标注
            img_id = img_info['id']
            annotations = seq_data['annotations'].get(img_id, [])
            
            if annotations:
                stats['frames_with_annotations'] += 1
                stats['total_annotations'] += len(annotations)
            
            # 绘制标注（优化：减少函数调用）
            for ann in annotations:
                drone_id = ann.get('drone_id', 1)
                stats['drone_ids'].add(drone_id)
                
                # 获取边界框 (COCO 格式: [x, y, width, height])
                x, y, w, h = ann['bbox']
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # 选择颜色
                color = self.colors.get(drone_id, (0, 255, 255))
                
                # 绘制边界框（使用更粗的线条，更明显）
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签（简化版本，减少绘制操作）
                label = f"D{drone_id}"  # 简化标签
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 添加信息文字（简化版本）
            info_text = f"Seq:{sequence_id} Frame:{idx + 1}/{len(images)} Time:{img_info['timestamp']:.1f}s Obj:{len(annotations)}"
            
            # 绘制信息面板（单行，更简洁）
            cv2.rectangle(frame, (10, 10), (600, 45), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (20, 32),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 写入视频（优化：批量写入可以提升性能，但OpenCV不支持，保持原样）
            if video_writer:
                video_writer.write(frame)
            
            # 显示窗口（仅在需要时）
            if show_window and idx % 2 == 0:  # 每2帧显示一次，减少窗口刷新
                cv2.imshow(f'Sequence {sequence_id} - {self.modality.upper()}', frame)
                
                # 按 'q' 退出，按 'p' 暂停
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户中断")
                    break
                elif key == ord('p'):
                    logger.info("暂停 - 按任意键继续")
                    cv2.waitKey(0)
        
        # 清理
        if video_writer:
            video_writer.release()
            file_size = video_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✓ 视频已保存: {video_file} ({file_size:.1f} MB)")
        
        if show_window:
            cv2.destroyAllWindows()
        
        # 完善统计信息
        stats['drone_ids'] = sorted(stats['drone_ids'])
        stats['annotation_rate'] = stats['frames_with_annotations'] / stats['total_frames'] if stats['total_frames'] > 0 else 0
        
        # 打印统计信息
        logger.info(f"\n{'='*70}")
        logger.info(f"序列 {sequence_id} 统计信息")
        logger.info(f"{'='*70}")
        logger.info(f"总帧数: {stats['total_frames']}")
        logger.info(f"有标注的帧: {stats['frames_with_annotations']} ({stats['annotation_rate']*100:.1f}%)")
        logger.info(f"总标注数: {stats['total_annotations']}")
        logger.info(f"平均标注/帧: {stats['total_annotations']/stats['total_frames']:.2f}")
        logger.info(f"Drone IDs: {stats['drone_ids']}")
        logger.info(f"{'='*70}\n")
        
        return stats
    
    def visualize_multiple_sequences(self, sequence_ids, export_video=False, 
                                    output_dir='visualizations', fps=30, 
                                    show_window=True, max_frames=None):
        """
        可视化多个序列
        
        Args:
            sequence_ids: 序列 ID 列表
            export_video: 是否导出视频
            output_dir: 输出目录
            fps: 视频帧率
            show_window: 是否显示窗口
            max_frames: 每个序列的最大帧数
        
        Returns:
            list: 每个序列的统计信息
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🎬 批量可视化 - 共 {len(sequence_ids)} 个序列")
        logger.info(f"{'='*70}")
        logger.info(f"序列列表: {sequence_ids}")
        logger.info(f"{'='*70}\n")
        
        all_stats = []
        
        for idx, seq_id in enumerate(sequence_ids, 1):
            logger.info(f"\n>>> 进度: [{idx}/{len(sequence_ids)}] 处理序列 {seq_id} <<<\n")
            stats = self.visualize_sequence(
                seq_id, 
                export_video=export_video,
                output_dir=output_dir,
                fps=fps,
                show_window=show_window,
                max_frames=max_frames
            )
            
            if stats:
                all_stats.append(stats)
        
        # 打印总体统计
        if all_stats:
            logger.info(f"\n{'='*70}")
            logger.info(f"总体统计 ({len(all_stats)} 个序列)")
            logger.info(f"{'='*70}")
            
            total_frames = sum(s['total_frames'] for s in all_stats)
            total_annotations = sum(s['total_annotations'] for s in all_stats)
            all_drone_ids = set()
            for s in all_stats:
                all_drone_ids.update(s['drone_ids'])
            
            logger.info(f"总帧数: {total_frames}")
            logger.info(f"总标注数: {total_annotations}")
            logger.info(f"平均标注/帧: {total_annotations/total_frames:.2f}")
            logger.info(f"所有 Drone IDs: {sorted(all_drone_ids)}")
            logger.info(f"{'='*70}\n")
        
        return all_stats
    
    def create_comparison_video(self, sequence_id, output_dir='visualizations', fps=30):
        """
        创建 RGB 和 Event 对比视频（需要两种模态都存在）
        
        Args:
            sequence_id: 序列 ID
            output_dir: 输出目录
            fps: 视频帧率
        
        Returns:
            str: 输出视频路径
        """
        logger.info(f"\n创建对比视频 - 序列 {sequence_id}")
        
        # 检查两种模态是否都存在
        rgb_visualizer = FREDSequenceVisualizer(self.fred_root, self.coco_root, 'rgb')
        event_visualizer = FREDSequenceVisualizer(self.fred_root, self.coco_root, 'event')
        
        if sequence_id not in rgb_visualizer.annotations:
            logger.error(f"RGB 模态中不存在序列 {sequence_id}")
            return None
        
        if sequence_id not in event_visualizer.annotations:
            logger.error(f"Event 模态中不存在序列 {sequence_id}")
            return None
        
        # 获取两种模态的图像
        rgb_images = rgb_visualizer.annotations[sequence_id]['images']
        event_images = event_visualizer.annotations[sequence_id]['images']
        
        # 使用较短的序列
        n_frames = min(len(rgb_images), len(event_images))
        
        logger.info(f"RGB 帧数: {len(rgb_images)}, Event 帧数: {len(event_images)}")
        logger.info(f"使用 {n_frames} 帧创建对比视频")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_file = output_path / f"sequence_{sequence_id}_comparison.mp4"
        
        # 读取第一帧以确定尺寸
        rgb_frame = cv2.imread(str(self.fred_root / rgb_images[0]['file_name']))
        event_frame = cv2.imread(str(self.fred_root / event_images[0]['file_name']))
        
        if rgb_frame is None or event_frame is None:
            logger.error("无法读取第一帧")
            return None
        
        h, w = rgb_frame.shape[:2]
        
        # 创建视频写入器（宽度翻倍）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_file), fourcc, fps, (w * 2, h))
        
        logger.info(f"导出对比视频: {video_file}")
        logger.info(f"分辨率: {w*2}x{h}, 帧率: {fps} FPS")
        
        # 处理每一帧
        logger.info(f"开始处理 {n_frames} 帧...")
        
        pbar = tqdm(range(n_frames), desc="对比视频",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for idx in pbar:
            # RGB 帧
            rgb_img_info = rgb_images[idx]
            rgb_path = self.fred_root / rgb_img_info['file_name']
            rgb_frame = cv2.imread(str(rgb_path))
            
            # Event 帧
            event_img_info = event_images[idx]
            event_path = self.fred_root / event_img_info['file_name']
            event_frame = cv2.imread(str(event_path))
            
            if rgb_frame is None or event_frame is None:
                continue
            
            # 更新进度
            if idx % 50 == 0:
                pbar.set_postfix({'时间': f"{rgb_img_info['timestamp']:.1f}s"})
            
            # 绘制 RGB 标注
            rgb_anns = rgb_visualizer.annotations[sequence_id]['annotations'].get(rgb_img_info['id'], [])
            for ann in rgb_anns:
                x, y, w_box, h_box = ann['bbox']
                drone_id = ann.get('drone_id', 1)
                color = self.colors.get(drone_id, (0, 255, 255))
                cv2.rectangle(rgb_frame, (int(x), int(y)), (int(x+w_box), int(y+h_box)), color, 2)
            
            # 绘制 Event 标注
            event_anns = event_visualizer.annotations[sequence_id]['annotations'].get(event_img_info['id'], [])
            for ann in event_anns:
                x, y, w_box, h_box = ann['bbox']
                drone_id = ann.get('drone_id', 1)
                color = self.colors.get(drone_id, (0, 255, 255))
                cv2.rectangle(event_frame, (int(x), int(y)), (int(x+w_box), int(y+h_box)), color, 2)
            
            # 添加标签
            cv2.putText(rgb_frame, "RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(event_frame, "EVENT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # 合并帧
            combined_frame = np.hstack([rgb_frame, event_frame])
            
            # 写入视频
            video_writer.write(combined_frame)
        
        video_writer.release()
        file_size = Path(video_file).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ 对比视频已保存: {video_file} ({file_size:.1f} MB)")
        
        return str(video_file)


def main():
    parser = argparse.ArgumentParser(
        description='FRED 数据集序列可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化单个序列并导出视频
  python visualize_fred_sequences.py --modality rgb --sequence 0 --export-video
  
  # 可视化随机序列（不导出视频）
  python visualize_fred_sequences.py --modality event --random
  
  # 可视化多个序列
  python visualize_fred_sequences.py --modality rgb --sequences 0 1 5 --export-video
  
  # 创建 RGB 和 Event 对比视频
  python visualize_fred_sequences.py --comparison --sequence 0
  
  # 快速预览（仅前100帧）
  python visualize_fred_sequences.py --modality rgb --sequence 0 --max-frames 100
  
  # 列出所有可用序列
  python visualize_fred_sequences.py --modality rgb --list-sequences
        """
    )
    
    parser.add_argument('--fred-root', type=str, 
                       default='/mnt/data/datasets/fred',
                       help='FRED 数据集根目录')
    parser.add_argument('--coco-root', type=str, 
                       default='datasets/fred_coco',
                       help='COCO 格式数据集根目录')
    parser.add_argument('--modality', type=str, 
                       default='rgb',
                       choices=['rgb', 'event'],
                       help='模态选择')
    parser.add_argument('--sequence', type=int,
                       help='序列 ID')
    parser.add_argument('--sequences', type=int, nargs='+',
                       help='多个序列 ID')
    parser.add_argument('--random', action='store_true',
                       help='随机选择一个序列')
    parser.add_argument('--export-video', action='store_true',
                       help='导出为视频文件')
    parser.add_argument('--output-dir', type=str, 
                       default='visualizations',
                       help='输出目录')
    parser.add_argument('--fps', type=int, default=30,
                       help='视频帧率')
    parser.add_argument('--no-window', action='store_true',
                       help='不显示窗口（仅导出视频）')
    parser.add_argument('--max-frames', type=int,
                       help='最大帧数（用于快速预览）')
    parser.add_argument('--list-sequences', action='store_true',
                       help='列出所有可用序列')
    parser.add_argument('--comparison', action='store_true',
                       help='创建 RGB 和 Event 对比视频')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = FREDSequenceVisualizer(
            fred_root=args.fred_root,
            coco_root=args.coco_root,
            modality=args.modality
        )
        
        # 列出所有序列
        if args.list_sequences:
            sequences = visualizer.get_available_sequences()
            logger.info(f"\n可用序列 ({len(sequences)} 个):")
            for seq_id in sequences:
                info = visualizer.get_sequence_info(seq_id)
                logger.info(f"  序列 {seq_id}: {info['n_images']} 帧, "
                          f"{info['n_annotations']} 标注, "
                          f"时长 {info['duration']:.1f}s, "
                          f"Drones: {info['drone_ids']}")
            return 0
        
        # 创建对比视频
        if args.comparison:
            if not args.sequence:
                logger.error("创建对比视频需要指定 --sequence")
                return 1
            
            visualizer.create_comparison_video(
                sequence_id=args.sequence,
                output_dir=args.output_dir,
                fps=args.fps
            )
            return 0
        
        # 确定要可视化的序列
        if args.sequences:
            sequence_ids = args.sequences
        elif args.sequence is not None:
            sequence_ids = [args.sequence]
        elif args.random:
            available = visualizer.get_available_sequences()
            if not available:
                logger.error("没有可用的序列")
                return 1
            sequence_ids = [random.choice(available)]
            logger.info(f"随机选择序列: {sequence_ids[0]}")
        else:
            logger.error("请指定 --sequence, --sequences, 或 --random")
            return 1
        
        # 可视化序列
        visualizer.visualize_multiple_sequences(
            sequence_ids=sequence_ids,
            export_video=args.export_video,
            output_dir=args.output_dir,
            fps=args.fps,
            show_window=not args.no_window,
            max_frames=args.max_frames
        )
        
        logger.info("\n✅ 可视化完成！")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
