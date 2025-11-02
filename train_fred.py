#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用FRED COCO数据集训练YOLOv5
"""
import datetime
import os
from functools import partial
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.callbacks_coco import CocoEvalCallback, SimplifiedEvalCallback
from utils.dataloader_coco import CocoYoloDataset, coco_dataset_collate
from utils.utils import (download_weights, get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

# 导入FRED配置
import config_fred as cfg

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='使用FRED COCO数据集训练YOLOv5')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='选择模态: rgb 或 event')
    parser.add_argument('--eval_only', action='store_true', 
                        help='只进行评估，不进行训练')
    parser.add_argument('--eval_map', action='store_true', default=True,
                        help='是否计算mAP（会增加训练时间）')
    parser.add_argument('--no_eval_map', action='store_true',
                        help='禁用mAP评估以加快训练速度')
    parser.add_argument('--resume', action='store_true',
                        help='从最佳权重继续训练')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速验证模式：仅运行100个batch验证功能正确性')
    args = parser.parse_args()
    
    # 如果指定了 --no_eval_map，则禁用mAP评估
    if args.no_eval_map:
        args.eval_map = False
    
    modality = args.modality
    
    # ========================================================================
    # 从配置文件加载参数
    # ========================================================================
    
    # 基本配置
    Cuda = cfg.CUDA
    seed = cfg.SEED
    distributed = cfg.DISTRIBUTED
    sync_bn = cfg.SYNC_BN
    fp16 = cfg.FP16
    
    # 模型配置
    input_shape = cfg.INPUT_SHAPE
    backbone = cfg.BACKBONE
    phi = cfg.PHI
    pretrained = cfg.PRETRAINED
    
    # 数据集路径配置
    train_json = cfg.get_annotation_path(modality, 'train')
    val_json = cfg.get_annotation_path(modality, 'val')
    test_json = cfg.get_annotation_path(modality, 'test')
    
    train_img_dir = cfg.get_image_dir(modality)
    val_img_dir = cfg.get_image_dir(modality)
    test_img_dir = cfg.get_image_dir(modality)
    
    # 检查文件是否存在
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"训练集标注文件不存在: {train_json}\n"
                              f"请先运行: python convert_fred_to_coco.py --modality {modality}")
    
    # 类别配置
    num_classes = cfg.NUM_CLASSES
    class_names = cfg.CLASS_NAMES
    
    # 先验框配置
    anchors_path = cfg.ANCHORS_PATH
    anchors_mask = cfg.ANCHORS_MASK
    
    # 模型权重（支持断点续练）
    if args.resume:
        model_path = cfg.get_model_path(modality, best=True)
        if not os.path.exists(model_path):
            print(f"警告: 未找到最佳权重 {model_path}，将从头训练")
            model_path = ''
    else:
        model_path = ''
    
    # 数据增强
    mosaic = cfg.MOSAIC
    mosaic_prob = cfg.MOSAIC_PROB
    mixup = cfg.MIXUP
    mixup_prob = cfg.MIXUP_PROB
    special_aug_ratio = cfg.SPECIAL_AUG_RATIO
    label_smoothing = cfg.LABEL_SMOOTHING
    
    # 训练参数
    Init_Epoch = cfg.INIT_EPOCH
    Freeze_Epoch = cfg.FREEZE_EPOCH
    UnFreeze_Epoch = cfg.UNFREEZE_EPOCH
    Freeze_batch_size = cfg.FREEZE_BATCH_SIZE
    Unfreeze_batch_size = cfg.UNFREEZE_BATCH_SIZE
    Freeze_Train = cfg.FREEZE_TRAIN
    
    # 优化器配置
    Init_lr = cfg.INIT_LR
    Min_lr = cfg.MIN_LR
    optimizer_type = cfg.OPTIMIZER_TYPE
    momentum = cfg.MOMENTUM
    weight_decay = cfg.WEIGHT_DECAY
    lr_decay_type = cfg.LR_DECAY_TYPE
    
    # 其他配置
    save_period = cfg.SAVE_PERIOD
    save_dir = cfg.get_save_dir(modality)
    eval_flag = cfg.EVAL_FLAG
    eval_period = cfg.EVAL_PERIOD
    num_workers = cfg.NUM_WORKERS
    prefetch_factor = cfg.PREFETCH_FACTOR
    persistent_workers = cfg.PERSISTENT_WORKERS
    
    # 快速验证模式
    if args.quick_test:
        print("\n" + "="*70)
        print("⚡ 快速验证模式")
        print("="*70)
        print("仅运行 2 个 epoch，每个 epoch 最多 100 个 batch")
        print("用于快速验证训练流程是否正确")
        print("="*70 + "\n")
        
        Init_Epoch = 0
        Freeze_Epoch = 1
        UnFreeze_Epoch = 2
        Freeze_Train = True
        eval_flag = True
        eval_period = 1
        save_period = 1
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 验证配置
    config_errors = cfg.validate_config()
    if config_errors:
        print("配置验证失败:")
        for error in config_errors:
            print(f"  - {error}")
        raise ValueError("配置错误，请检查 config_fred.py")
    
    seed_everything(seed)
    
    # 设置设备
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
    
    # 获取先验框
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone, phi)
            dist.barrier()
        else:
            download_weights(backbone, phi)
    
    # 创建模型
    model = YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
    if not pretrained:
        weights_init(model)
    
    if model_path != '' and os.path.exists(model_path):
        if local_rank == 0:
            print(f'加载权重: {model_path}')
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print(f"成功加载的键数量: {len(load_key)}")
            print(f"未加载的键数量: {len(no_load_key)}")
    
    # 损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    
    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    
    # 混合精度训练
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    model_train = model.train()
    
    # 多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    # 权值平滑
    ema = ModelEMA(model_train)
    
    # 创建数据集
    print(f"\n加载FRED {modality.upper()}数据集...")
    print(f"  训练集: {train_json}")
    print(f"  验证集: {val_json}")
    print(f"  测试集: {test_json}")
    
    train_dataset = CocoYoloDataset(
        train_json, train_img_dir, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup,
        mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True,
        special_aug_ratio=special_aug_ratio
    )
    
    val_dataset = CocoYoloDataset(
        val_json, val_img_dir, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=UnFreeze_Epoch, mosaic=False, mixup=False,
        mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0
    )
    
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    
    if local_rank == 0:
        print(f"\n数据集统计:")
        print(f"  训练集: {num_train} 张图片")
        print(f"  验证集: {num_val} 张图片")
        print(f"  类别数: {num_classes}")
        print(f"  模态: {modality.upper()}")
    
    if local_rank == 0:
        show_config(
            model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
            optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            num_train=num_train, num_val=num_val
        )
        
        # 检查训练步数
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(f"\n\033[1;33;44m[Warning] 使用{optimizer_type}优化器时，建议将训练总步长设置到{wanted_step}以上。\033[0m")
            print(f"\033[1;33;44m[Warning] 本次运行的总训练数据量为{num_train}，Unfreeze_batch_size为{Unfreeze_batch_size}，共训练{UnFreeze_Epoch}个Epoch，计算出总训练步长为{total_step}。\033[0m")
            print(f"\033[1;33;44m[Warning] 由于总训练步长为{total_step}，小于建议总步长{wanted_step}，建议设置总世代为{wanted_epoch}。\033[0m")
    
    # 开始训练
    if True:
        UnFreeze_flag = False
        
        # 冻结训练
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        
        # 自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        # 优化器
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        
        # 学习率调度器
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        # 计算epoch步数
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        if ema:
            ema.updates = epoch_step * Init_Epoch
        
        # 数据加载器
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, 
                        num_workers=num_workers, pin_memory=True, drop_last=True,
                        collate_fn=coco_dataset_collate, sampler=train_sampler,
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                        prefetch_factor=prefetch_factor if num_workers > 0 else None,
                        persistent_workers=persistent_workers if num_workers > 0 else False)
        
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            collate_fn=coco_dataset_collate, sampler=val_sampler,
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                            prefetch_factor=prefetch_factor if num_workers > 0 else None,
                            persistent_workers=persistent_workers if num_workers > 0 else False)
        
        # 评估回调（使用测试集）
        if local_rank == 0:
            # 快速验证模式：限制评估样本数量
            max_eval_samples = 100 if args.quick_test else None
            
            # 根据命令行参数决定是否使用完整的mAP评估
            if args.eval_map and eval_flag:
                eval_callback = CocoEvalCallback(
                    net=model,
                    input_shape=input_shape,
                    anchors=anchors,
                    anchors_mask=anchors_mask,
                    class_names=class_names,
                    num_classes=num_classes,
                    coco_json_path=test_json,
                    image_dir=test_img_dir,
                    log_dir=log_dir,
                    cuda=Cuda,
                    map_out_path=os.path.join(save_dir, ".temp_map_out"),
                    max_boxes=cfg.MAX_BOXES,
                    confidence=cfg.CONFIDENCE,
                    nms_iou=cfg.NMS_IOU,
                    letterbox_image=cfg.LETTERBOX_IMAGE,
                    MINOVERLAP=cfg.MINOVERLAP,
                    eval_flag=eval_flag,
                    period=eval_period,
                    max_eval_samples=max_eval_samples
                )
                print("\n✓ 使用COCO格式的mAP评估（会增加训练时间）")
                print(f"  - 评估周期: 每 {eval_period} 个epoch")
                print(f"  - 评估数据集: 测试集 ({test_json})")
                if max_eval_samples:
                    print(f"  - ⚡ 快速验证: 仅评估 {max_eval_samples} 个样本")
            else:
                # 使用简化版回调（只记录epoch，不计算mAP）
                eval_callback = SimplifiedEvalCallback(log_dir, eval_flag=False, period=1)
                print("\n注意: mAP评估已禁用（使用 --no_eval_map 禁用，默认启用）")
                print("  - 如需加快训练速度，可使用: python train_fred.py --no_eval_map")
        else:
            eval_callback = SimplifiedEvalCallback(log_dir, eval_flag=False, period=1)
        
        # 训练循环
        if args.eval_only:
            print("\n仅评估模式（功能待实现）")
        else:
            print(f"\n开始训练 - {modality.upper()}模态")
            print("=" * 70)
            
            for epoch in range(Init_Epoch, UnFreeze_Epoch):
                # 解冻训练
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size = Unfreeze_batch_size
                    
                    nbs = 64
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                    
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                    for param in model.backbone.parameters():
                        param.requires_grad = True
                    
                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size
                    
                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                    
                    if ema:
                        ema.updates = epoch_step * epoch
                    
                    if distributed:
                        batch_size = batch_size // ngpus_per_node
                    
                    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=True, drop_last=True,
                                    collate_fn=coco_dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                                    persistent_workers=persistent_workers if num_workers > 0 else False)
                    
                    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size,
                                        num_workers=num_workers, pin_memory=True, drop_last=True,
                                        collate_fn=coco_dataset_collate, sampler=val_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                                        prefetch_factor=prefetch_factor if num_workers > 0 else None,
                                        persistent_workers=persistent_workers if num_workers > 0 else False)
                    
                    UnFreeze_flag = True
                
                gen.dataset.epoch_now = epoch
                gen_val.dataset.epoch_now = epoch
                
                if distributed:
                    train_sampler.set_epoch(epoch)
                
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                
                # 快速验证模式：限制batch数量
                max_batches = 100 if args.quick_test else None
                fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback,
                            optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                            UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank, max_batches=max_batches)
                
                if distributed:
                    dist.barrier()
            
            if local_rank == 0:
                loss_history.writer.close()
                
                # 保存最终模型
                final_model_path = cfg.get_model_path(modality, best=False)
                torch.save(model.state_dict(), final_model_path)
                print(f"\n训练完成！最终模型已保存: {final_model_path}")
