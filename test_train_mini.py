#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用mini数据集快速测试训练
只需要几分钟即可完成
"""
import datetime
import os
from functools import partial
import argparse

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
    parser = argparse.ArgumentParser(description='使用mini数据集快速测试训练')
    parser.add_argument('--modality', type=str, default='rgb', 
                       choices=['rgb', 'event'],
                       help='选择模态')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮次')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"快速训练测试 - {args.modality.upper()}模态 - {args.epochs} epochs")
    print("=" * 80)
    
    # 基本配置
    modality = args.modality
    Cuda = True
    seed = 11
    fp16 = False
    
    # 模型配置
    input_shape = [640, 640]
    backbone = 'cspdarknet'
    phi = 's'
    pretrained = False  # 快速测试不使用预训练
    
    # Mini数据集配置
    mini_root = f'datasets/fred_mini/{modality}'
    
    # 检查mini数据集是否存在
    if not os.path.exists(mini_root):
        print(f"\n错误: Mini数据集不存在: {mini_root}")
        print(f"\n请先创建mini数据集:")
        print(f"  /home/yz/miniforge3/envs/torch/bin/python3 create_mini_dataset.py --modality {modality}")
        exit(1)
    
    train_json = os.path.join(mini_root, 'annotations', 'instances_train.json')
    val_json = os.path.join(mini_root, 'annotations', 'instances_val.json')
    test_json = os.path.join(mini_root, 'annotations', 'instances_test.json')
    
    train_img_dir = os.path.join(mini_root, 'train')
    val_img_dir = os.path.join(mini_root, 'val')
    test_img_dir = os.path.join(mini_root, 'test')
    
    # 类别配置
    num_classes = 1
    class_names = ['object']
    
    # 先验框配置
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # 训练参数（快速测试）
    Init_Epoch = 0
    UnFreeze_Epoch = args.epochs
    batch_size = 4  # 小batch
    Freeze_Train = False  # 不冻结
    
    # 优化器配置
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 0
    lr_decay_type = "cos"
    
    # 数据增强（关闭以加快速度）
    mosaic = False
    mixup = False
    label_smoothing = 0
    
    # 其他配置
    save_dir = f'logs/test_mini_{modality}'
    num_workers = 2
    eval_flag = False  # 快速测试不评估mAP
    
    os.makedirs(save_dir, exist_ok=True)
    
    seed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0
    
    print(f"\n配置:")
    print(f"  模态: {modality.upper()}")
    print(f"  设备: {device}")
    print(f"  训练轮次: {UnFreeze_Epoch}")
    print(f"  Batch size: {batch_size}")
    
    # 获取先验框
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 创建模型
    print(f"\n创建模型...")
    model = YoloBody(anchors_mask, num_classes, phi, backbone, 
                    pretrained=pretrained, input_shape=input_shape)
    weights_init(model)
    print(f"✓ 模型创建成功")
    
    # 损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, 
                        anchors_mask, label_smoothing)
    
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
    print(f"\n加载mini数据集...")
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
    
    # 计算epoch步数
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    print(f"\nEpoch步数:")
    print(f"  训练: {epoch_step} steps")
    print(f"  验证: {epoch_step_val} steps")
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("样本数太少，请增加mini数据集大小")
    
    if ema:
        ema.updates = 0
    
    # 数据加载器
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                    num_workers=num_workers, pin_memory=True, drop_last=True,
                    collate_fn=coco_dataset_collate,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, drop_last=True,
                        collate_fn=coco_dataset_collate,
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    
    # 评估回调
    eval_callback = SimplifiedEvalCallback(log_dir, eval_flag=False, period=1)
    
    # 开始训练
    print("\n" + "=" * 80)
    print(f"开始训练 - {args.epochs} epochs")
    print("=" * 80)
    
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(
            model_train, model, ema, yolo_loss, loss_history, eval_callback,
            optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
            UnFreeze_Epoch, Cuda, fp16, scaler, 1, save_dir, local_rank
        )
    
    loss_history.writer.close()
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, f'mini_{modality}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    print("\n" + "=" * 80)
    print("✅ 快速训练测试完成！")
    print("=" * 80)
    print(f"模型保存: {final_model_path}")
    print(f"日志目录: {log_dir}")
    print(f"\n可以开始正式训练:")
    print(f"  /home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality {modality}")
