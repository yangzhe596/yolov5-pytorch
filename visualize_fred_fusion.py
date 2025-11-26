#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED Fusion 数据集可视化脚本
将融合数据集中的配对帧生成视频，可视化标注边界框

输出：
- fusion_rgb.mp4: RGB 模态 + 标注的视频
- fusion_event.mp4: Event 模态 + 标注的视频
- fusion_comparison.mp4: RGB 和 Event 并排对比的视频
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from datetime import datetime

# 配置和颜色
COLORS = {
    'dual_rgb': (0, 255, 0),      # 绿色 - 双模态（RGB）
    'dual_event': (255, 255, 0),  # 青色 - 双模态（Event）
    'rgb_only': (0, 165, 255),    # 橙色 - 仅 RGB
    'event_only': (255, 0, 0),    # 红色 - 仅 Event
}
FRED_ROOT = Path('/mnt/data/datasets/fred')
DEFAULT_OUTPUT = Path('/mnt/data/code/yolov5-pytorch/results/fusion_visualization')


class FusionVisualizer:
    """FRED Fusion 数据集可视化器"""
    
    def __init__(self, annotation_path, output_dir=DEFAULT_OUTPUT, 
                 fps=30, max_frames=None, modality='dual',
                 draw_rgb=True, draw_event=True):
        """
        初始化可视化器
        
        Args:
            annotation_path: COCO JSON 标注文件路径
            output_dir: 输出目录
            fps: 视频帧率
            max_frames: 最大处理帧数（None 表示全部）
            modality: 可视化模态 'rgb', 'event', 'dual'
            draw_rgb: 是否绘制 RGB 边界框
            draw_event: 是否绘制 Event 边界框
        """
        self.annotation_path = Path(annotation_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames
        self.modality = modality
        self.draw_rgb = draw_rgb
        self.draw_event = draw_event
        
        # 加载标注数据
        print(f"加载标注文件: {self.annotation_path}")
        with open(self.annotation_path, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']
        self.image_to_annotations = self._build_annotation_map()
        
        print(f"数据统计:")
        print(f"  - 总图片数: {len(self.images)}")
        print(f"  - 总标注数: {len(self.annotations)}")
        print(f"  - 类别: {[c['name'] for c in self.categories]}")
        
        # 计算模态统计
        modality_counts = {}
        for img in self.images:
            mod = img.get('modality', 'unknown')
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        print(f"  - 模态分布:")
        for mod, count in modality_counts.items():
            print(f"    * {mod}: {count}")
    
    def _build_annotation_map(self):
        """构建图像到标注的映射"""
        img_to_ann = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in img_to_ann:
                img_to_ann[img_id] = []
            img_to_ann[img_id].append(ann)
        return img_to_ann
    
    def get_image_path(self, img_info, modality='rgb'):
        """获取图片完整路径"""
        if modality == 'rgb':
            subpath = Path(img_info['rgb_file_name'])
            dataset_type = subpath.parts[0] if subpath.parts else '0'
            base_dir = FRED_ROOT / dataset_type / 'PADDED_RGB'
            # 获取文件名（去除序列号前缀）
            filename = subpath.name
            # 查找实际文件
            return base_dir / filename
        elif modality == 'event':
            subpath = Path(img_info['event_file_name'])
            dataset_type = subpath.parts[0] if subpath.parts else '0'
            base_dir = FRED_ROOT / dataset_type / 'Event/Frames'
            filename = subpath.name
            return base_dir / filename
        return None
    
    def check_image_exists(self, img_info, modality='rgb'):
        """检查图片是否存在"""
        path = self.get_image_path(img_info, modality)
        if path and path.exists():
            return path
        return None
    
    def convert_event_to_rgb(self, event_img):
        """将 Event 图片转换为伪彩色"""
        if len(event_img.shape) == 2:  # 灰度图
            # 使用 jet colormap 转换为伪彩色
            event_rgb = cv2.applyColorMap(event_img, cv2.COLORMAP_JET)
            return event_rgb
        elif len(event_img.shape) == 3 and event_img.shape[2] == 3:
            return event_img
        elif len(event_img.shape) == 3 and event_img.shape[2] == 1:
            event_rgb = cv2.applyColorMap(event_img, cv2.COLORMAP_JET)
            return event_rgb
        return event_img
    
    def draw_annotations(self, image, annotations, modality='rgb'):
        """在图像上绘制边界框"""
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = [int(v) for v in bbox]
            
            # 确定颜色
            ann_modality = ann.get('modality', 'unknown')
            if ann_modality == 'dual':
                color = COLORS['dual_rgb'] if modality == 'rgb' else COLORS['dual_event']
            elif ann_modality == 'rgb':
                color = COLORS['rgb_only']
            elif ann_modality == 'event':
                color = COLORS['event_only']
            else:
                color = (0, 255, 255)  # 黄色
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            category_id = ann['category_id']
            category_name = next(c['name'] for c in self.categories if c['id'] == category_id)
            label = f"{category_name} ({ann_modality})"
            
            # 计算标签背景大小
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image, 
                (x, y - label_height - 10), 
                (x + label_width, y),
                color, 
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                image, 
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return image
    
    def add_metadata(self, image, img_info, modality='rgb'):
        """添加元信息到图像"""
        h, w = image.shape[:2]
        
        # 添加时间信息
        rgb_time = img_info.get('rgb_timestamp', 0)
        event_time = img_info.get('event_timestamp', 0)
        time_diff = img_info.get('time_diff', 0)
        fusion_status = img_info.get('fusion_status', 'unknown')
        
        # 标题栏背景
        cv2.rectangle(image, (0, 0), (w, 40), (0, 0, 0), -1)
        
        # 添加文本
        texts = [
            f"Modality: {modality.upper()}",
            f"Fusion: {fusion_status}",
            f"RGB Time: {rgb_time:.3f}s",
            f"Event Time: {event_time:.3f}s",
            f"Diff: {time_diff*1000:.2f}ms"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                image,
                text,
                (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        return image
    
    def visualize_rgb(self):
        """生成 RGB 模态可视化视频"""
        if not self.draw_rgb:
            return
        
        print("\n" + "="*60)
        print("生成 RGB 模态可视化视频")
        print("="*60)
        
        # 创建视频写入器
        video_path = self.output_dir / 'fusion_rgb.mp4'
        writer = None
        
        count = 0
        for img_info in tqdm(self.images, desc="处理 RGB 帧"):
            if self.max_frames and count >= self.max_frames:
                break
            
            # 检查切换 modality 过滤
            if self.modality != 'dual' and img_info.get('modality') != self.modality:
                continue
            
            # 获取 RGB 图片路径
            rgb_path = self.check_image_exists(img_info, 'rgb')
            if not rgb_path:
                continue
            
            # 读取图像
            image = cv2.imread(str(rgb_path))
            if image is None:
                continue
            
            # 初始化视频写入器
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (w, h)
                )
            
            # 绘制边界框
            img_id = img_info['id']
            anns = self.image_to_annotations.get(img_id, [])
            image = self.draw_annotations(image, anns, modality='rgb')
            
            # 添加元信息
            image = self.add_metadata(image, img_info, modality='rgb')
            
            # 写入视频
            writer.write(image)
            count += 1
        
        if writer:
            writer.release()
            print(f"✅ RGB 视频已保存: {video_path}")
            print(f"   帧数: {count}, FPS: {self.fps}")
        else:
            print("❌ 未找到可用的 RGB 图片")
    
    def visualize_event(self):
        """生成 Event 模态可视化视频"""
        if not self.draw_event:
            return
        
        print("\n" + "="*60)
        print("生成 Event 模态可视化视频")
        print("="*60)
        
        # 创建视频写入器
        video_path = self.output_dir / 'fusion_event.mp4'
        writer = None
        
        count = 0
        for img_info in tqdm(self.images, desc="处理 Event 帧"):
            if self.max_frames and count >= self.max_frames:
                break
            
            # 检查切换 modality 过滤
            if self.modality != 'dual' and img_info.get('modality') != self.modality:
                continue
            
            # 获取 Event 图片路径
            event_path = self.check_image_exists(img_info, 'event')
            if not event_path:
                continue
            
            # 读取图像
            image = cv2.imread(str(event_path))
            if image is None:
                continue
            
            # 转换为伪彩色
            image = self.convert_event_to_rgb(image)
            
            # 初始化视频写入器
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (w, h)
                )
            
            # 绘制边界框
            img_id = img_info['id']
            anns = self.image_to_annotations.get(img_id, [])
            image = self.draw_annotations(image, anns, modality='event')
            
            # 添加元信息
            image = self.add_metadata(image, img_info, modality='event')
            
            # 写入视频
            writer.write(image)
            count += 1
        
        if writer:
            writer.release()
            print(f"✅ Event 视频已保存: {video_path}")
            print(f"   帧数: {count}, FPS: {self.fps}")
        else:
            print("❌ 未找到可用的 Event 图片")
    
    def visualize_comparison(self):
        """生成 RGB 和 Event 对比视频"""
        print("\n" + "="*60)
        print("生成 RGB vs Event 对比视频")
        print("="*60)
        
        video_path = self.output_dir / 'fusion_comparison.mp4'
        writer = None
        
        count = 0
        for img_info in tqdm(self.images, desc="处理对比帧"):
            if self.max_frames and count >= self.max_frames:
                break
            
            # 检查切换 modality 过滤
            if self.modality != 'dual' and img_info.get('modality') != self.modality:
                continue
            
            # 获取 RGB 和 Event 图像
            rgb_path = self.check_image_exists(img_info, 'rgb')
            event_path = self.check_image_exists(img_info, 'event')
            
            if not rgb_path or not event_path:
                continue
            
            # 读取图像
            rgb_img = cv2.imread(str(rgb_path))
            event_img = cv2.imread(str(event_path))
            
            if rgb_img is None or event_img is None:
                continue
            
            # 转换 Event 为伪彩色
            event_img = self.convert_event_to_rgb(event_img)
            
            # 调整尺寸（取最小尺寸）
            h = min(rgb_img.shape[0], event_img.shape[0])
            w = min(rgb_img.shape[1], event_img.shape[1])
            rgb_img = cv2.resize(rgb_img, (w, h))
            event_img = cv2.resize(event_img, (w, h))
            
            # 水平拼接
            combined = np.hstack([rgb_img, event_img])
            
            # 初始化视频写入器
            if writer is None:
                ch, cw = combined.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (cw, ch)
                )
            
            # 绘制边界框（两边都绘制）
            img_id = img_info['id']
            anns = self.image_to_annotations.get(img_id, [])
            
            # 在左半部分（RGB）绘制
            rgb_with_ann = self.draw_annotations(rgb_img.copy(), anns, modality='rgb')
            rgb_with_ann = self.add_metadata(rgb_with_ann, img_info, modality='rgb')
            
            # 在右半部分（Event）绘制
            event_with_ann = self.draw_annotations(event_img.copy(), anns, modality='event')
            event_with_ann = self.add_metadata(event_with_ann, img_info, modality='event')
            
            # 拼接
            combined_ann = np.hstack([rgb_with_ann, event_with_ann])
            
            # 添加分隔线
            cv2.line(
                combined_ann,
                (w, 0),
                (w, h),
                (255, 255, 255),
                2
            )
            
            # 写入视频
            writer.write(combined_ann)
            count += 1
        
        if writer:
            writer.release()
            print(f"✅ 对比视频已保存: {video_path}")
            print(f"   帧数: {count}, FPS: {self.fps}")
        else:
            print("❌ 未找到可用的对比帧")
    
    def run(self):
        """运行可视化"""
        print("\n" + "="*80)
        print("FRED Fusion 数据集可视化")
        print("="*80)
        print(f"输入: {self.annotation_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"模态: {self.modality}")
        print(f"FPS: {self.fps}")
        print(f"最大帧数: {self.max_frames if self.max_frames else '全部'}")
        print("="*80)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成视频
        if self.modality in ['dual', 'rgb']:
            self.visualize_rgb()
        
        if self.modality in ['dual', 'event']:
            self.visualize_event()
        
        if self.modality == 'dual':
            self.visualize_comparison()


def main():
    parser = argparse.ArgumentParser(
        description='FRED Fusion 数据集可视化脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有视频（RGB, Event, 对比）
  python visualize_fred_fusion.py \\
    --annotation datasets/fred_fusion_test/annotations/instances_val.json

  # 只生成 RGB 视频
  python visualize_fred_fusion.py \\
    --annotation datasets/fred_fusion_test/annotations/instances_val.json \\
    --modality rgb

  # 限制帧数为 100 帧
  python visualize_fred_fusion.py \\
    --annotation datasets/fred_fusion_test/annotations/instances_val.json \\
    --max-frames 100

  # 设置输出帧率为 60 FPS
  python visualize_fred_fusion.py \\
    --annotation datasets/fred_fusion_test/annotations/instances_val.json \\
    --fps 60
        """
    )
    
    parser.add_argument(
        '--annotation',
        type=str,
        default='datasets/fred_fusion_test/annotations/instances_val.json',
        help='COCO 标注文件路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help='输出目录'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='视频帧率'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='最大处理帧数（默认：全部）'
    )
    
    parser.add_argument(
        '--modality',
        type=str,
        choices=['dual', 'rgb', 'event'],
        default='dual',
        help='可视化模态：dual（全部）, rgb, event'
    )
    
    parser.add_argument(
        '--no-rgb',
        action='store_true',
        help='不生成 RGB 视频'
    )
    
    parser.add_argument(
        '--no-event',
        action='store_true',
        help='不生成 Event 视频'
    )
    
    args = parser.parse_args()
    
    # 检查标注文件
    annotation_path = Path(args.annotation)
    if not annotation_path.exists():
        print(f"❌ 标注文件不存在: {annotation_path}")
        return
    
    # 创建可视化器
    visualizer = FusionVisualizer(
        annotation_path=annotation_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        modality=args.modality,
        draw_rgb=not args.no_rgb,
        draw_event=not args.no_event
    )
    
    # 运行可视化
    visualizer.run()
    
    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)


if __name__ == '__main__':
    main()