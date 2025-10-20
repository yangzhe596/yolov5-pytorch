#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练一个epoch
验证训练流程是否正常
"""
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.callbacks_coco import SimplifiedEvalCallback
from utils.dataloader_coco import CocoYoloDataset, coco_dataset_collate
from utils.utils import (get_anchors, seed_everything, worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    print("=" * 80)
    print("FRED数据集训练测试 - 1个Epoch")
    print("=" * 80)
    
    # 基本配置
    modality = 'rgb'  # 测试RGB模态
    Cuda = True
    seed = 11
    fp16 = False
    
    # 模型配置
    input_shape = [640, 640]
    backbone = 'cspdarknet'
    phi = 's'
    pretrained = False  # 测试时不使用预训练权重，加快速度
    
    # FRED数据集配置
    coco_root = f'datasets/fred_coco/{modality}'
    train_json = os.path.join(coco_root, 'annotations', 'instances_train.json')
    val_json = os.path.join(coco_root, 'annotations', 'instances_val.json')
    train_img_dir = os.path.join(coco_root, 'train')
    val_img_dir = os.path.join(coco_root, 'val')
    
    # 类别配置
    num_classes = 1
    
    # 先验框配置
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # 训练参数（测试用，只训练1个epoch）
    Init_Epoch = 0
    UnFreeze_Epoch = 1  # 只训练1个epoch
    batch_size = 4  # 小batch size加快测试
    Freeze_Train = False  # 不冻结，直接训练
    
    # 优化器配置
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"  # 使用adam加快收敛
    momentum = 0.937
    weight_decay = 0  # adam不使用weight_decay
    lr_decay_type = "cos"
    
    # 数据增强（测试时关闭）
    mosaic = False
    mixup = False
    label_smoothing = 0
    
    # 其他配置
    save_dir = 'logs/test_fred'
    num_workers = 2
    
    os.makedirs(save_dir, exist_ok=True)
    
    seed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0
    
    print(f"\n配置信息:")
    print(f"  模态: {modality.upper()}")
    print(f"  设备: {device}")
    print(f"  训练轮次: {UnFreeze_Epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  输入尺寸: {input_shape}")
    
    # 获取先验框
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 创建模型
    print(f"\n创建模型...")
    model = YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
    weights_init(model)
    print(f"✓ 模型创建成功")
    
    # 损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    
    # 记录Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    
    scaler = None
    
    model_train = model.train()
    
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    
    # 权值平滑
    ema = ModelEMA(model_train)
    
    # 创建数据集
    print(f"\n加载数据集...")
    train_dataset = CocoYoloDataset(
        train_json, train_img_dir, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup,
        mosaic_prob=0, mixup_prob=0, train=True, special_aug_ratio=0
    )
    
    val_dataset = CocoYoloDataset(
        val_json, val_img_dir, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=UnFreeze_Epoch, mosaic=False, mixup=False,
        mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0
    )
    
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    
    print(f"✓ 数据集加载完成")
    print(f"  训练集: {num_train} 张")
    print(f"  验证集: {num_val} 张")
    
    # 优化器
    print(f"\n配置优化器...")
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    optimizer = optim.Adam(pg0, Init_lr, betas=(momentum, 0.999))
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
    
    print(f"✓ 优化器配置完成")
    
    # 计算epoch步数
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    print(f"\nEpoch步数:")
    print(f"  训练: {epoch_step} steps")
    print(f"  验证: {epoch_step_val} steps")
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小或batch_size过大")
    
    if ema:
        ema.updates = 0
    
    # 数据加载器
    print(f"\n创建数据加载器...")
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                    num_workers=num_workers, pin_memory=True, drop_last=True,
                    collate_fn=coco_dataset_collate,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, drop_last=True,
                        collate_fn=coco_dataset_collate,
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    
    print(f"✓ 数据加载器创建完成")
    
    # 使用简化版的eval_callback
    eval_callback = SimplifiedEvalCallback(log_dir, eval_flag=False, period=1)
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练测试（1个epoch）")
    print("=" * 80)
    
    epoch = 0
    gen.dataset.epoch_now = epoch
    gen_val.dataset.epoch_now = epoch
    
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    
    try:
        fit_one_epoch(
            model_train, model, ema, yolo_loss, loss_history, eval_callback,
            optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
            UnFreeze_Epoch, Cuda, fp16, scaler, 1, save_dir, local_rank
        )
        
        print("\n" + "=" * 80)
        print("✅ 训练测试成功！")
        print("=" * 80)
        print(f"\n训练日志保存在: {log_dir}")
        print(f"\n可以开始正式训练:")
        print(f"  /home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 训练测试失败！")
        print("=" * 80)
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if loss_history:
            loss_history.writer.close()
