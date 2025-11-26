#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRED Fusion 数据集可视化结果检查脚本
显示已生成的视频文件和统计信息
"""

import os
from pathlib import Path
import subprocess

# 配置
RESULTS_DIR = Path('/mnt/data/code/yolov5-pytorch/results/fusion_visualization/preview')

def get_video_info(video_path):
    """获取视频文件信息"""
    if not video_path.exists():
        return None
    
    # 使用 ffprobe 获取视频信息
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
            if video_stream:
                duration = float(info['format'].get('duration', 0))
                size = int(info['format'].get('size', 0))
                bitrate = int(info['format'].get('bit_rate', 0))
                return {
                    'duration': duration,
                    'size_mb': size / (1024 * 1024),
                    'bitrate': bitrate,
                    'resolution': f"{video_stream['width']}x{video_stream['height']}",
                    'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                    'frames': int(video_stream.get('nb_frames', 0))
                }
    except:
        pass
    
    # 备用：使用 stat
    stat = video_path.stat()
    return {
        'size_mb': stat.st_size / (1024 * 1024),
        'frames': 'N/A',
    }

def main():
    print("\n" + "="*80)
    print("FRED Fusion 数据集可视化结果")
    print("="*80)
    print(f"目录: {RESULTS_DIR}\n")
    
    if not RESULTS_DIR.exists():
        print("❌ 目录不存在")
        return
    
    # 列出视频文件
    video_files = list(RESULTS_DIR.glob('*.mp4'))
    image_files = list(RESULTS_DIR.glob('*.jpg'))
    
    if not video_files and not image_files:
        print("❌ 未找到视频或图像文件")
        return
    
    # 显示图像文件
    if image_files:
        print("📸 静态图像预览:")
        print("-" * 80)
        for img in sorted(image_files):
            stat = img.stat()
            print(f"  {img.name:40s} {stat.st_size / 1024:6.1f} KB")
        print()
    
    # 显示视频文件
    if video_files:
        print("🎬 视频文件:")
        print("-" * 80)
        for video in sorted(video_files):
            info = get_video_info(video)
            if info:
                if 'resolution' in info:
                    print(f"  {video.name:40s}")
                    print(f"      尺寸: {info['size_mb']:.1f} MB")
                    print(f"      分辨率: {info['resolution']}")
                    print(f"      时长: {info['duration']:.1f} 秒")
                    print(f"      FPS: {info['fps']}")
                    print(f"      帧数: {info['frames']}")
                else:
                    print(f"  {video.name:40s} {info['size_mb']:.1f} MB")
            else:
                print(f"  {video.name:40s} (信息读取失败)")
        print()
    
    # 统计
    total_size = 0
    for f in RESULTS_DIR.glob('*'):
        if f.is_file():
            total_size += f.stat().st_size
    
    print("="*80)
    print(f"总计: {len(video_files)} 个视频, {len(image_files)} 个图像")
    print(f"总大小: {total_size / (1024 * 1024):.1f} MB")
    print("="*80)
    print()

if __name__ == '__main__':
    main()