#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv5 训练性能基准测试
测试数据加载、数据增强、模型推理、反向传播等关键步骤的耗时

使用方法:
    python benchmark/training_benchmark.py --modality rgb --num_batches 100
    python benchmark/training_benchmark.py --modality event --num_batches 50 --batch_size 16
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_fred as cfg
from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss
from utils.dataloader_coco import CocoYoloDataset, coco_dataset_collate
from utils.utils import get_anchors, worker_init_fn


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
        print("\n" + "=" * 80)
        print("性能基准测试结果")
        print("=" * 80)
        
        # 按平均时间排序
        sorted_names = sorted(self.times.keys(), 
                            key=lambda x: np.mean(self.times[x]), 
                            reverse=True)
        
        print(f"\n{'步骤':<20} {'平均(ms)':<12} {'标准差(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'总计(s)':<12}")
        print("-" * 80)
        
        total_time = 0
        for name in sorted_names:
            stats = self.get_stats(name)
            if stats:
                print(f"{name:<20} {stats['mean']*1000:>10.2f}  {stats['std']*1000:>10.2f}  "
                      f"{stats['min']*1000:>10.2f}  {stats['max']*1000:>10.2f}  {stats['total']:>10.2f}")
                total_time += stats['total']
        
        print("-" * 80)
        print(f"{'总计':<20} {'':<12} {'':<12} {'':<12} {'':<12} {total_time:>10.2f}")
        
        # 计算百分比
        print(f"\n{'步骤':<20} {'占比(%)':<12} {'吞吐量(samples/s)':<20}")
        print("-" * 80)
        
        for name in sorted_names:
            stats = self.get_stats(name)
            if stats and total_time > 0:
                percentage = (stats['total'] / total_time) * 100
                throughput = stats['count'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{name:<20} {percentage:>10.2f}  {throughput:>18.2f}")


class BenchmarkDataLoader:
    """带计时功能的 DataLoader 包装器"""
    def __init__(self, dataloader, stats, name_prefix=""):
        self.dataloader = dataloader
        self.stats = stats
        self.name_prefix = name_prefix
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        self.stats.start(f"{self.name_prefix}data_loading")
        try:
            batch = next(self.iterator)
        finally:
            self.stats.end(f"{self.name_prefix}data_loading")
        return batch
    
    def __len__(self):
        return len(self.dataloader)


def benchmark_training(args):
    """训练性能基准测试"""
    print("=" * 80)
    print("YOLOv5 训练性能基准测试")
    print("=" * 80)
    print(f"模态: {args.modality}")
    print(f"Batch Size: {args.batch_size}")
    print(f"测试批次数: {args.num_batches}")
    print(f"输入尺寸: {cfg.INPUT_SHAPE}")
    print(f"主干网络: {cfg.BACKBONE}")
    print(f"模型版本: YOLOv5-{cfg.PHI}")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 创建计时器
    stats = TimingStats()
    
    # ========================================================================
    # 1. 加载数据集
    # ========================================================================
    print("\n[1/5] 加载数据集...")
    stats.start("dataset_init")
    
    train_json = cfg.get_annotation_path(args.modality, 'train')
    train_img_dir = cfg.get_image_dir(args.modality)
    
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"训练集标注文件不存在: {train_json}")
    
    anchors, num_anchors = get_anchors(cfg.ANCHORS_PATH)
    
    train_dataset = CocoYoloDataset(
        train_json, train_img_dir, cfg.INPUT_SHAPE, cfg.NUM_CLASSES, 
        anchors, cfg.ANCHORS_MASK,
        epoch_length=cfg.UNFREEZE_EPOCH, 
        mosaic=cfg.MOSAIC, mixup=cfg.MIXUP,
        mosaic_prob=cfg.MOSAIC_PROB, mixup_prob=cfg.MIXUP_PROB, 
        train=True,
        special_aug_ratio=cfg.SPECIAL_AUG_RATIO
    )
    
    stats.end("dataset_init")
    print(f"  ✓ 数据集大小: {len(train_dataset)}")
    
    # ========================================================================
    # 2. 创建 DataLoader
    # ========================================================================
    print("\n[2/5] 创建 DataLoader...")
    stats.start("dataloader_init")
    
    dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=coco_dataset_collate,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=cfg.SEED)
    )
    
    stats.end("dataloader_init")
    print(f"  ✓ DataLoader 配置:")
    print(f"    - num_workers: {args.num_workers}")
    print(f"    - prefetch_factor: {args.prefetch_factor if args.num_workers > 0 else 'N/A'}")
    print(f"    - persistent_workers: {args.persistent_workers if args.num_workers > 0 else False}")
    print(f"    - pin_memory: True")
    
    # ========================================================================
    # 3. 创建模型
    # ========================================================================
    print("\n[3/5] 创建模型...")
    stats.start("model_init")
    
    model = YoloBody(cfg.ANCHORS_MASK, cfg.NUM_CLASSES, cfg.PHI, 
                     cfg.BACKBONE, pretrained=False, input_shape=cfg.INPUT_SHAPE)
    model = model.to(device)
    model.train()
    
    stats.end("model_init")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ 模型参数:")
    print(f"    - 总参数: {total_params:,}")
    print(f"    - 可训练参数: {trainable_params:,}")
    
    # ========================================================================
    # 4. 创建损失函数和优化器
    # ========================================================================
    print("\n[4/5] 创建损失函数和优化器...")
    
    yolo_loss = YOLOLoss(anchors, cfg.NUM_CLASSES, cfg.INPUT_SHAPE, 
                         args.cuda, cfg.ANCHORS_MASK, cfg.LABEL_SMOOTHING)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.INIT_LR, 
                               momentum=cfg.MOMENTUM, nesterov=True)
    
    print(f"  ✓ 损失函数: YOLOLoss")
    print(f"  ✓ 优化器: SGD (lr={cfg.INIT_LR}, momentum={cfg.MOMENTUM})")
    
    # ========================================================================
    # 5. 运行基准测试
    # ========================================================================
    print("\n[5/5] 运行基准测试...")
    print(f"  测试 {args.num_batches} 个批次...")
    
    # 预热
    print("\n  预热中 (5 batches)...")
    warmup_loader = iter(dataloader)
    for _ in range(min(5, len(dataloader))):
        try:
            images, bboxes, y_trues = next(warmup_loader)
            if args.cuda:
                images = images.cuda()
            
            with torch.no_grad():
                outputs = model(images)
        except StopIteration:
            break
    
    # 同步 CUDA
    if args.cuda:
        torch.cuda.synchronize()
    
    print("  开始基准测试...")
    
    # 包装 DataLoader
    benchmark_loader = BenchmarkDataLoader(dataloader, stats, "train_")
    
    batch_count = 0
    pbar = tqdm(total=args.num_batches, desc="  进度")
    
    for batch_data in benchmark_loader:
        if batch_count >= args.num_batches:
            break
        
        images, bboxes, y_trues = batch_data
        
        # 数据传输到 GPU
        stats.start("train_data_transfer")
        if args.cuda:
            images = images.cuda()
            y_trues = [y.cuda() for y in y_trues]
        if args.cuda:
            torch.cuda.synchronize()
        stats.end("train_data_transfer")
        
        # 前向传播
        stats.start("train_forward")
        outputs = model(images)
        if args.cuda:
            torch.cuda.synchronize()
        stats.end("train_forward")
        
        # 计算损失
        stats.start("train_loss_compute")
        loss_value = 0
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], bboxes, y_trues[l])
            loss_value += loss_item
        if args.cuda:
            torch.cuda.synchronize()
        stats.end("train_loss_compute")
        
        # 反向传播
        stats.start("train_backward")
        optimizer.zero_grad()
        loss_value.backward()
        if args.cuda:
            torch.cuda.synchronize()
        stats.end("train_backward")
        
        # 优化器更新
        stats.start("train_optimizer_step")
        optimizer.step()
        if args.cuda:
            torch.cuda.synchronize()
        stats.end("train_optimizer_step")
        
        batch_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # ========================================================================
    # 6. 打印结果
    # ========================================================================
    stats.print_summary()
    
    # 额外的性能指标
    print("\n" + "=" * 80)
    print("额外性能指标")
    print("=" * 80)
    
    # 计算每个 batch 的总时间
    batch_times = []
    for i in range(batch_count):
        batch_time = 0
        for key in stats.times.keys():
            if key.startswith("train_") and i < len(stats.times[key]):
                batch_time += stats.times[key][i]
        if batch_time > 0:
            batch_times.append(batch_time)
    
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        samples_per_sec = args.batch_size / avg_batch_time
        
        print(f"\n平均每批次时间: {avg_batch_time*1000:.2f} ms")
        print(f"训练吞吐量: {samples_per_sec:.2f} samples/s")
        print(f"预计每 epoch 时间: {len(train_dataset) / samples_per_sec / 60:.2f} 分钟")
        print(f"预计 {cfg.UNFREEZE_EPOCH} epochs 总时间: {len(train_dataset) / samples_per_sec * cfg.UNFREEZE_EPOCH / 3600:.2f} 小时")
    
    # GPU 内存使用
    if args.cuda:
        print(f"\nGPU 内存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # 瓶颈分析
    print("\n" + "=" * 80)
    print("瓶颈分析")
    print("=" * 80)
    
    total_time = sum(stats.get_stats(key)['total'] for key in stats.times.keys() if stats.get_stats(key))
    
    bottlenecks = []
    for key in stats.times.keys():
        stat = stats.get_stats(key)
        if stat:
            percentage = (stat['total'] / total_time) * 100
            bottlenecks.append((key, percentage, stat['mean']))
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    print("\n主要耗时步骤 (占比 > 5%):")
    for name, percentage, avg_time in bottlenecks:
        if percentage > 5:
            print(f"  - {name}: {percentage:.1f}% (平均 {avg_time*1000:.2f} ms)")
    
    # 优化建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    
    data_loading_time = stats.get_stats("train_data_loading")
    forward_time = stats.get_stats("train_forward")
    
    if data_loading_time and forward_time:
        data_ratio = data_loading_time['total'] / total_time * 100
        compute_ratio = forward_time['total'] / total_time * 100
        
        print(f"\n数据加载占比: {data_ratio:.1f}%")
        print(f"模型计算占比: {compute_ratio:.1f}%")
        
        if data_ratio > 20:
            print("\n⚠️  数据加载成为瓶颈，建议:")
            print("  1. 增加 num_workers (当前: {})".format(args.num_workers))
            print("  2. 增加 prefetch_factor (当前: {})".format(args.prefetch_factor))
            print("  3. 使用更快的存储设备 (SSD)")
            print("  4. 减少数据增强的复杂度")
        
        if compute_ratio > 60:
            print("\n⚠️  模型计算成为瓶颈，建议:")
            print("  1. 使用混合精度训练 (FP16)")
            print("  2. 增加 batch_size (当前: {})".format(args.batch_size))
            print("  3. 使用更小的模型 (当前: YOLOv5-{})".format(cfg.PHI))
    
    print("\n" + "=" * 80)
    print("基准测试完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 训练性能基准测试')
    
    # 数据集参数
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                       help='选择模态: rgb 或 event')
    
    # 测试参数
    parser.add_argument('--num_batches', type=int, default=100,
                       help='测试的批次数量 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小 (默认: 使用配置文件中的值)')
    
    # DataLoader 参数
    parser.add_argument('--num_workers', type=int, default=None,
                       help='DataLoader workers 数量 (默认: 使用配置文件中的值)')
    parser.add_argument('--prefetch_factor', type=int, default=None,
                       help='预取因子 (默认: 使用配置文件中的值)')
    parser.add_argument('--persistent_workers', type=bool, default=None,
                       help='是否使用持久化 workers (默认: 使用配置文件中的值)')
    
    # 设备参数
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='是否使用 CUDA (默认: True)')
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                       help='禁用 CUDA')
    
    args = parser.parse_args()
    
    # 使用配置文件中的默认值
    if args.batch_size is None:
        args.batch_size = cfg.UNFREEZE_BATCH_SIZE
    if args.num_workers is None:
        args.num_workers = cfg.NUM_WORKERS
    if args.prefetch_factor is None:
        args.prefetch_factor = cfg.PREFETCH_FACTOR
    if args.persistent_workers is None:
        args.persistent_workers = cfg.PERSISTENT_WORKERS
    
    # 运行基准测试
    benchmark_training(args)


if __name__ == '__main__':
    main()
