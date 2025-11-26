#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRED Fusion 数据集训练脚本 - 优化版本
针对训练速度慢的问题进行优化
"""
import argparse
import datetime
import os
import sys
import time
from copy import deepcopy as copy
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo_fusion import YoloFusionBody
from nets.yolo_training import ModelEMA, YOLOLoss, get_lr_scheduler
from utils.callbacks import LossHistory
from utils.callbacks_coco import CocoEvalCallback
from utils.dataloader_fusion import FusionYoloDataset, fusion_dataset_collate
from utils.utils import (get_anchors, get_classes, get_lr, seed_everything,
                         show_config, worker_init_fn)

# 导入 FRED 配置
import config_fred

def weights_init(net, init_type='normal', init_gain=0.02):
    """网络权重初始化"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    
    print(f'Initializing network with {init_type} init')
    net.apply(init_func)

class SimplifiedEvalCallback:
    """简化版评估回调：不计算 mAP"""
    def __init__(self, log_dir, eval_flag=False, period=1):
        self.log_dir = log_dir
        self.eval_flag = eval_flag
        self.period = period
        self.best_map = 0
    
    def on_epoch_end(self, epoch, model):
        """最后时更新最佳模型"""
        pass

def fit_one_epoch_fusion(model_train, model, ema, yolo_loss, loss_history, eval_callback,
                         optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                         UnFreeze_Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank, max_batches=None):
    """Fusion 模型专用的训练函数"""
    try:
        loss        = 0
        val_loss    = 0
        total_dual  = 0
        total_time_diff = 0
        
        if local_rank == 0:
            print('\n' + '-'*80)
            start_time = time.time()
        
        model_train.train()
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3, disable=local_rank != 0) as pbar:
            for iteration, batch in enumerate(gen):
                if max_batches and iteration >= max_batches:
                    break
                
                if iteration >= epoch_step:
                    break
                
                # 获取数据
                rgb_images, event_images = batch[0]
                targets = batch[1]
                y_trues = batch[2]
                fusion_infos = batch[3]
                
                # 计算融合统计
                if fusion_infos:
                    for info in fusion_infos:
                        total_dual += 1 if info['fusion_status'] == 'dual' else 0
                        total_time_diff += abs(info['time_diff'])
                
                with torch.no_grad():
                    if cuda:
                        # 分别将两个图像移动到 GPU
                        rgb_images = rgb_images.cuda(local_rank, non_blocking=True)  # 使用 non_blocking
                        event_images = event_images.cuda(local_rank, non_blocking=True)
                        targets = [ann.cuda(local_rank, non_blocking=True) for ann in targets]
                        y_trues = [ann.cuda(local_rank, non_blocking=True) for ann in y_trues]
                
                # 清零梯度
                optimizer.zero_grad()
                
                if not fp16:
                    # 前向传播
                    outputs = model_train(rgb_images, event_images)
                    
                    loss_value_all  = 0
                    # 计算损失
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                        loss_value_all  += loss_item
                
                    # 反向传播
                    loss_value_all.backward()
                    optimizer.step()
                    
                    if ema:
                        ema.update(model_train)
                        
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # 前向传播
                        outputs = model_train(rgb_images, event_images)
                        
                        loss_value_all  = 0
                        # 计算损失
                        for l in range(len(outputs)):
                            loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                            loss_value_all  += loss_item
                    
                    # 反向传播
                    scaler.scale(loss_value_all).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if ema:
                        ema.update(model_train)
                
                loss += loss_value_all.item()
                
                if local_rank == 0:
                    pbar.set_postfix(**{
                        'loss'  : loss / (iteration + 1),
                        'lr'    : get_lr(optimizer),
                        'dual_rate': f'{total_dual}/{max(1, (iteration + 1) * len(rgb_images))} ({total_dual/max(1, (iteration + 1) * len(rgb_images)):.1%})',
                        'avg_tdiff': f'{total_time_diff/max(1, total_dual) * 1000:.2f}ms' if total_dual > 0 else 'N/A'
                    })
                    pbar.update(1)
        
        if local_rank == 0:
            # 显示融合统计
            total_samples = epoch_step * len(rgb_images) if epoch_step > 0 else 0
            if total_samples > 0:
                print(f'融合信息: Dual Rate: {total_dual}/{total_samples} ({total_dual/total_samples:.1%}), '
                      f'Avg Time Diff: {total_time_diff/max(1, total_dual) * 1000:.2f}ms')
            
            train_time = time.time() - start_time
            print(f'Train cost time: {train_time:.1f}s')
        
        # 验证集验证
        if local_rank == 0:
            print('\n' + '-'*80)
            start_time = time.time()
        
        model_train.eval()
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3, disable=local_rank != 0) as pbar:
            for iteration, batch in enumerate(gen_val):
                if max_batches and iteration >= max_batches:
                    break
                
                if iteration >= epoch_step_val:
                    break
                
                # 获取数据
                rgb_images, event_images = batch[0]
                targets = batch[1]
                y_trues = batch[2]
                
                with torch.no_grad():
                    if cuda:
                        # 分别将两个图像移动到 GPU
                        rgb_images = rgb_images.cuda(local_rank, non_blocking=True)
                        event_images = event_images.cuda(local_rank, non_blocking=True)
                        targets = [ann.cuda(local_rank, non_blocking=True) for ann in targets]
                        y_trues = [ann.cuda(local_rank, non_blocking=True) for ann in y_trues]
                    
                    # 前向传播
                    outputs = model_train(rgb_images, event_images)
                    
                    loss_value_all  = 0
                    # 计算损失
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                        loss_value_all  += loss_item
            
                    val_loss += loss_value_all.item()
                
                if local_rank == 0:
                    pbar.set_postfix(**{
                        'val_loss': val_loss / (iteration + 1)
                    })
                    pbar.update(1)
        
        # 更新验证集损失曲线
        if local_rank == 0:
            val_time = time.time() - start_time
            print(f'Val cost time: {val_time:.1f}s')
            
            loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
            
            # EMA 模型评估
            eval_model = ema.ema if ema else model
            
            if eval_callback:
                eval_callback.on_epoch_end(epoch + 1, eval_model)
            
            # 保存模型
            if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
                if ema:
                    save_state_dict = ema.ema.state_dict()
                else:
                    save_state_dict = model.state_dict()
                
                try:
                    torch.save({'state_dict': save_state_dict}, os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
                except:
                    pass
            
            if ema:
                save_model_state = ema.ema
            else:
                save_model_state = model
            
            loss_history.best_model_wts = copy(save_model_state.state_dict())
            
        print('-'*80 + '\n')
    
    except Exception as e:
        print(f'训练过程中发生错误: {e}')
        raise

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """设置优化器的学习率"""
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    # 全局配置
    globals().update(vars(config_fred))
    
    # 初始化
    seed_everything(1)
    
    # 获取训练配置
    cfg = config_fred
    
    # === 核心配置 ===
    
    # 显卡配置
    Cuda            = True
    fp16            = True  # 优化：启用混合精度训练

    # 数据集和类别配置
    num_classes     = config_fred.NUM_CLASSES
    class_names     = config_fred.CLASS_NAMES
    anchors_path    = 'model_data/yolo_anchors.txt'
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='FRED Fusion 数据集训练 - 优化版本')
    parser.add_argument('--modality', type=str, default='dual', choices=['dual', 'rgb', 'event'],
                        help='训练模态: dual(双模态), rgb, event')
    parser.add_argument('--compression_ratio', type=float, default=0.75,
                        help='融合压缩比率 (0.25~1.0), 推荐 0.75')
    parser.add_argument('--freeze_training', action='store_true',
                        help='仅进行冻结训练（不进行解冻训练）')
    parser.add_argument('--high_res', action='store_true',
                        help='启用高分辨率模式')
    parser.add_argument('--no_eval_map', action='store_true',
                        help='禁用 mAP 评估，加快训练速度')
    parser.add_argument('--resume', action='store_true',
                        help='从最佳权重继续训练')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速验证模式：仅运行100个batch验证功能正确性')
    parser.add_argument('--eval_only', action='store_true',
                        help='只进行评估，不训练')
    args = parser.parse_args()
    
    # 默认评估模式
    default_eval_mode = True
    
    if args.no_eval_map:
        print("\n⚠️  注意: mAP 评估已禁用（使用 --no_eval_map 禁用，默认启用）")
        default_eval_mode = False
        eval_callback = SimplifiedEvalCallback(cfg.get_save_dir('fusion'), eval_flag=False, period=1)
    else:
        # 使用 COCO 格式的 mAP 评估（需要 JSON 文件）
        print("\n✓ 使用COCO格式的mAP评估（会增加训练时间）")
    
    # 设置训练模态
    fusion_modality = args.modality
    
    # 设置压缩比率
    compression_ratio = args.compression_ratio
    
    # Fusion 标注文件路径
    train_json = config_fred.get_fusion_annotation_path('train')
    val_json   = config_fred.get_fusion_annotation_path('val')
    test_json  = config_fred.get_fusion_annotation_path('test')
    
    # Fusion 模型保存目录
    save_dir = config_fred.get_save_dir('fusion')
    
    # 高分辨率模式配置
    high_res = args.high_res
    
    # 模型配置
    backbone    = 'cspdarknet'
    phi         = 's'
    
    # 训练配置
    Init_Epoch          = 0
    Freezed_Epoch       = 50
    UnFreeze_Epoch      = 300
    
    eval_period         = 5  # 每 5 个 epoch 评估一次 mAP
    save_period         = 10 # 每 10 个 epoch 保存一次模型
    
    Freeze_batch_size   = 16
    Unfreeze_batch_size = 8
    
    # 优化器和学习率
    optimizer_type      = 'sgd'
    Init_lr             = 1e-2
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = 'cos'
    
    # 最小学习率
    Min_lr = Init_lr * 0.01
    
    # mAP 评估配置
    max_boxes           = 300
    confidence          = 0.001
    nms_iou             = 0.5
    letterbox_image     = True
    MINOVERLAP          = 0.5
    max_eval_samples    = 10000  # 最大评估样本数
    
    # === 优化：减少 workers 数量 ===
    num_workers         = 2  # 优化：从 4 降低到 2
    prefetch_factor     = 2  # 优化：从 4 降低到 2
    persistent_workers  = False  # 优化：每个 epoch 重新创建 workers
    
    # 数据增强（优化：降低概率）
    mosaic              = True
    mosaic_prob         = 0.3  # 优化：从 0.5 降低
    mixup               = True
    mixup_prob          = 0.3  # 优化：从 0.5 降低
    
    # 冻结训练
    Freeze_Train        = True
    
    # 简化模式 - 快速验证
    if args.quick_test:
        print("\n⚡ 快速验证模式: 仅运行100个batch")
        max_batches = 100
    else:
        max_batches = None
    
    # 调试模式 - 只评估
    if args.eval_only:
        max_batches = 10
    
    # 下载预训练权重
    if not args.resume:
        # 使用 RGB 模态的模型作为预训练权重（通常效果最好）
        model_path = cfg.get_model_path('rgb', best=True)
        if not os.path.exists(model_path):
            print(f"警告: RGB 预训练权重不存在 {model_path}")
            print("将使用默认的预训练权重路径")
            model_path = f"model_data/yolov5{phi}.pth"
    else:
        model_path = 'logs/fred_fusion/fred_fusion_best.pth'
    
    # 获取 anchors
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 设置网络输入尺寸和先验框
    if not high_res:
        input_shape = [640, 640]
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    else:
        input_shape = [640, 640]
        anchors_path = 'model_data/yolo_anchors_high_res.txt'
        anchors, num_anchors = get_anchors(anchors_path)
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # === 数据集准备 ===
    
    print("\n" + "="*80)
    print("Fusion 数据集准备")
    print("="*80)
    
    # 检查数据集
    # 计算 epoch 长度
    train_num = len(open(train_json, 'r').read().split('images')[0].split('"id"')) if os.path.exists(train_json) else 0
    val_num = len(open(val_json, 'r').read().split('images')[0].split('"id"')) if os.path.exists(val_json) else 0
    
    # 创建 Fusion 数据集（使用融合标注）
    fusion_train_set = FusionYoloDataset(
        train_json, fred_root, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=train_num // Freeze_batch_size, 
        mosaic=True, mixup=True, mosaic_prob=cfg.MOSAIC_PROB, mixup_prob=cfg.MIXUP_PROB, train=True,
        modality=fusion_modality, use_fusion_info=True
    ) if not args.eval_only else None
    
    fusion_val_set = FusionYoloDataset(
        val_json, fred_root, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=val_num // Unfreeze_batch_size,
        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
        modality='rgb', use_fusion_info=True
    )
    
    # === 构建训练和验证数据加载器 ===
    
    if not args.eval_only:
        gen = DataLoader(
            fusion_train_set, shuffle=True, batch_size=Freeze_batch_size, 
            num_workers=num_workers, pin_memory=True,  # 优化：使用非阻塞传输
            drop_last=True, collate_fn=fusion_dataset_collate,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        
        gen_val = DataLoader(
            fusion_val_set, shuffle=True, batch_size=Unfreeze_batch_size, 
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=fusion_dataset_collate,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        
        epoch_step = min(len(gen), 10 if args.quick_test else len(gen))
        epoch_step_val = min(len(gen_val), 10 if args.quick_test else len(gen_val))
        
        # 如果只有验证集
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请检查数据集")
    else:
        # 评估模式
        gen_val = DataLoader(
            fusion_val_set, shuffle=True, batch_size=Unfreeze_batch_size, 
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=fusion_dataset_collate,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        
        epoch_step_val = len(gen_val)
        epoch_step = 0
    
    # === 创建模型 ===
    
    print("\n" + "="*80)
    print("创建 Fusion 模型")
    print("="*80)
    
    model = YoloFusionBody(
        anchors_mask=anchors_mask,
        num_classes=num_classes,
        compression_ratio=compression_ratio,
        phi=phi,
        backbone=backbone,
        high_res=high_res
    )
    
    if not args.eval_only:
        weights_init(model)
    if model_path != '':
        print(f'载入预训练权重: {model_path}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        
        # Fusion 模型有双流 backbone，需要特殊处理
        fusion_pretrained = {}
        for k, v in pretrained_dict.items():
            # 如果是 backbone 权重，复制一份用于 RGB 和 Event 流
            if k.startswith('backbone.'):
                # RGB 流
                rgb_key = f'backbone_rgb.{k[9:]}'
                if rgb_key in model_dict and np.shape(model_dict[rgb_key]) == np.shape(v):
                    fusion_pretrained[rgb_key] = v
                
                # Event 流（使用相同的权重初始化）
                event_key = f'backbone_event.{k[9:]}'
                if event_key in model_dict and np.shape(model_dict[event_key]) == np.shape(v):
                    fusion_pretrained[event_key] = v
            # 其他权重直接使用
            elif k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                fusion_pretrained[k] = v
        
        print(f"成功加载 {len(fusion_pretrained)} 个权重参数到 Fusion 模型")
        model_dict.update(fusion_pretrained)
        model.load_state_dict(model_dict, strict=False)
    
    # 使用多 GPU
    if Cuda and torch.cuda.device_count() > 1 and not args.eval_only:
        print(f'使用 {torch.cuda.device_count()} 个 GPU 训练')
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model.cuda()
    
    # === 优化器和损失函数 ===
    
    if not args.eval_only:
        # 创建优化器
        optimizer = {
            'adam'  : optim.Adam(model_train.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model_train.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]
        
        # 学习率衰减
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
        
        # 损失函数
        yolo_loss   = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing=0)
        
        # 记录训练过程
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
        
        # 创建 scaler (用于 mixed precision training)
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    else:
        optimizer = None
        yolo_loss = None
        loss_history = None
        scaler = None
    
    # === EMA (指数移动平均) ===
    
    if not args.eval_only:
        ema = ModelEMA(model_train)
    else:
        ema = None
    
    # === 显示配置 ===
    
    print("\n" + "="*80)
    print("训练配置 - 优化版本")
    print("="*80)
    
    show_config(
        modality=args.modality,
        compression_ratio=compression_ratio,
        model_path=model_path,
        input_shape=input_shape,
        Init_Epoch=Init_Epoch,
        Freeze_Epoch=Freezed_Epoch if Freeze_Train else 0,
        UnFreeze_Epoch=UnFreeze_Epoch if Freeze_Train else 0,
        Freeze_batch_size=Freeze_batch_size,
        Unfreeze_batch_size=Unfreeze_batch_size,
        Freeze_Train=(Freeze_Train and not args.freeze_training and not args.quick_test),
        Init_lr=Init_lr,
        Min_lr=Min_lr,
        optimizer_type=optimizer_type,
        momentum=momentum,
        lr_decay_type=lr_decay_type,
        save_period=save_period,
        save_dir=save_dir,
        num_workers=num_workers,  # 优化：显示新的 workers 数量
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_train=len(gen) if not args.eval_only else 0,
        num_val=len(gen_val) if not args.eval_only else 0,
        high_res=high_res,
        anchors_path=anchors_path,
        anchors_mask=anchors_mask,
    )
    
    # 显示优化信息
    if not args.eval_only:
        print("\n" + "="*80)
        print("性能优化配置")
        print("="*80)
        print(f"✓ 混合精度训练 (fp16): {'启用' if fp16 else '禁用'}")
        print(f"✓ Workers 数量: {num_workers} (从 4 降低到 2)")
        print(f"✓ Prefetch 因子: {prefetch_factor} (从 4 降低到 2)")
        print(f"✓ Persistent workers: {persistent_workers} (每个 epoch 重新创建)")
        print(f"✓ Mosaic 概率: {mosaic_prob} (从 0.5 降低到 0.3)")
        print(f"✓ Mixup 概率: {mixup_prob} (从 0.5 降低到 0.3)")
        print(f"✓ mAP 评估: {'启用' if default_eval_mode else '禁用'}")
        print("="*80)
    
    if args.eval_only:
        print("-"*80)
        print("⚠️  评估模式：只进行验证集评估，不进行训练")
        print("-"*80)
    
    # === 冻结训练阶段 ===
    
    if not args.eval_only:
        if Freeze_Train and not args.freeze_training and not args.quick_test:
            print("\n" + "="*80)
            print("第一阶段：冻结训练 (0-%d epoch)" % Freezed_Epoch)
            print("="*80)
            print("  - 冻结主干网络，只训练检测头")
            print("  - 显存占用较小")
            print("  - 适合：显存不足、快速收敛")
            
            # 冻结主干网络
            for param in model.backbone_rgb.parameters():
                param.requires_grad = False
            for param in model.backbone_event.parameters():
                param.requires_grad = False
            for param in model.compression_convs.parameters():
                param.requires_grad = False
            
            # 优化器优化所有未冻结的参数
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_train.parameters()), 
                                 Init_lr, momentum=momentum, weight_decay=weight_decay)
            
            # 重新创建 scaler 以适应新的 optimizer
            scaler = torch.cuda.amp.GradScaler(enabled=fp16)
            
            # 训练参数
            batch_size = Freeze_batch_size
            UnFreeze_Epoch = Freezed_Epoch
            
            # 学习率调整
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
            
            # 如果是从头开始训练（没有预训练权重）
            if model_path == '':
                batch_size = 2
                Init_Epoch = 0
                UnFreeze_Epoch = 100
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_Epoch, Min_lr, UnFreeze_Epoch)
            
            # 开始训练
            for epoch in range(Init_Epoch, Freezed_Epoch):
                gen.dataset.mosaic = True
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                
                fit_one_epoch_fusion(model_train, model, ema, yolo_loss, loss_history, None,
                                     optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                                     UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, 0, max_batches=max_batches)
                
                if args.quick_test:
                    break
    
    # === 解冻训练阶段 ===
    
    if not args.eval_only:
        if args.freeze_training:
            print("\n" + "="*80)
            print("仅进行冻结训练，跳过解冻训练阶段")
            print("="*80)
        elif args.quick_test:
            print("\n" + "="*80)
            print("快速验证完成")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("第二阶段：解冻训练 (%d-%d epoch)" % (Freezed_Epoch, UnFreeze_Epoch))
            print("="*80)
            print("  - 解冻主干网络，全网络训练")
            print("  - 显存占用较大")
            print("  - 适合：追求最佳性能")
            
            # 解冻所有层
            for param in model.parameters():
                param.requires_grad = True
            
            # 定义优化器
            optimizer = {
                'adam'  : optim.Adam(model_train.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
                'sgd'   : optim.SGD(model_train.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
            }[optimizer_type]
            
            # 学习率调整
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
            
            batch_size = Unfreeze_batch_size
            
            # 开始训练
            for epoch in range(Freezed_Epoch, UnFreeze_Epoch):
                gen.dataset.mosaic = True
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                
                fit_one_epoch_fusion(model_train, model, ema, yolo_loss, loss_history, None,
                                     optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                                     UnFreeze_Epoch, Cuda, fp16, None, save_period, save_dir, 0, max_batches=max_batches)
                
                if args.quick_test:
                    break
    
    # === 仅评估模式 ===
    
    if args.eval_only:
        print("\n" + "="*80)
        print("开始评估（仅验证集）")
        print("="*80)
        
        if default_eval_mode:
            eval_callback.on_epoch_end(1, model)
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print("\n模型保存位置: " + save_dir)
    print("日志保存位置: " + save_dir)
    print("\n最佳模型: logs/fred_fusion/fred_fusion_best.pth")
    print("最终模型: logs/fred_fusion/fred_fusion_final.pth")
    print("\n" + "="*80)