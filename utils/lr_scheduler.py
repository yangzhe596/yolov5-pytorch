#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学习率调度工具
支持warmup和多种衰减策略
"""
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    """
    带warmup的余弦学习率衰减
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, warmup_start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

class WarmupMultiStepLR(_LRScheduler):
    """
    带warmup的多阶段学习率衰减
    """
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, warmup_start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_start_lr = warmup_start_lr
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # 多阶段衰减
            lr = self.base_lrs
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    lr = [l * self.gamma for l in lr]
            return lr

def create_lr_scheduler(optimizer, lr_decay_type, total_epochs, warmup_epochs=5, 
                       min_lr=1e-5, warmup_start_lr=1e-6, milestones=None, gamma=0.5):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        lr_decay_type: 'cos' 或 'step'
        total_epochs: 总训练轮次
        warmup_epochs: warmup轮次
        min_lr: 最小学习率
        warmup_start_lr: warmup起始学习率
        milestones: 多阶段衰减的节点（仅用于step模式）
        gamma: 衰减系数（仅用于step模式）
    
    Returns:
        学习率调度器
    """
    if lr_decay_type == 'cos':
        return WarmupCosineLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs,
            min_lr=min_lr,
            warmup_start_lr=warmup_start_lr
        )
    elif lr_decay_type == 'step':
        if milestones is None:
            milestones = [int(total_epochs * 0.6), int(total_epochs * 0.8)]
        return WarmupMultiStepLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            milestones=milestones,
            gamma=gamma,
            warmup_start_lr=warmup_start_lr
        )
    else:
        raise ValueError(f"不支持的学习率衰减类型: {lr_decay_type}")

def adjust_learning_rate(optimizer, epoch, lr_schedule):
    """
    手动调整学习率（可选）
    
    Args:
        optimizer: 优化器
        epoch: 当前轮次
        lr_schedule: 学习率调度器
    """
    if lr_schedule is not None:
        lr_schedule.step()
        current_lr = optimizer.param_groups[0]['lr']
        return current_lr
    return None