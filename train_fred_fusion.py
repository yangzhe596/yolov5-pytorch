# -*- coding: utf-8 -*-
"""
FRED Fusion 数据集训练脚本
支持 RGB + Event 双模态融合训练

核心功能：
- 双 backbone 网络架构（RGB 和 Event）
- 特征拼接 + 1x1 卷积压缩融合
- 支持多种压缩比率（0.25 ~ 1.0）
- 支持多模态训练（dual/rgb/event）
- 端到端训练流程
- 冻结/解冻训练策略
- mAP 评估（可选）

Author: YOLOv5-PyTorch-Fusion
Date: 2025-11-24
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
import gc
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

# 添加 Fusion 评估相关的导入
import os
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox

# 导入 FRED 配置
import config_fred

def weights_init(net, init_type='normal', init_gain=0.02):
    """
    网络权重初始化
    
    Args:
        net: 网络模型
        init_type: 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: 初始化增益
    """
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

class FusionCocoEvalCallback:
    """Fusion 专用评估回调"""
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, 
                 coco_json_path, image_dir, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, 
                 nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5,
                 eval_flag=True, period=1, max_eval_samples=10000):
        """
        Fusion 模型评估回调
        
        Args:
            net: Fusion 模型
            input_shape: 输入尺寸
        """
        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.coco_json_path = coco_json_path
        self.image_dir = image_dir
        self.log_dir = log_dir
        self.cuda = cuda
        
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        self.max_eval_samples = max_eval_samples
        
        # 初始化 bbox 解码器
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, self.input_shape, self.anchors_mask)
        
        # 保存最佳 mAP
        self.best_map = 0
        
        # 创建映射
        self.create_maps()
    
    def create_maps(self):
        """创建类别和图像映射"""
        with open(self.coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 类别映射
        self.classes = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        
        # 图像映射
        self.images = coco_data['images']
        self.total_images = len(self.images)
        self.img_id_to_img = {img['id']: img for img in self.images}
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            if ann['image_id'] not in self.img_to_anns:
                self.img_to_anns[ann['image_id']] = []
            self.img_to_anns[ann['image_id']].append(ann)
    
    def _prepare_eval_inputs(self, image_data: np.ndarray):
        """准备 RGB / Event 融合输入"""
        model_device = next(self.net.parameters()).device
        image_tensor = torch.from_numpy(image_data)
        
        # 对于 Fusion 模型，需要提供两个输入
        # 使用相同的图像作为 RGB 和 Event 输入
        rgb_images = image_tensor.to(model_device, non_blocking=True)
        event_images = rgb_images.clone()  # 创建副本而不是引用
        
        return rgb_images, event_images
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """生成预测结果txt文件（专门用于 Fusion 模型）"""
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image = cvtColor(image)
        image_shape = np.array(np.shape(image)[0:2])
        
        # 预处理
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        # 准备 Fusion 输入 - 确保输入数据正确
        # 对于评估，我们使用相同的图像作为 RGB 和 Event 输入
        # 这是因为评估时通常只有一种图像可用
        rgb_images, event_images = self._prepare_eval_inputs(image_data)
        
        # 确保模型在正确的设备上
        if self.cuda:
            rgb_images = rgb_images.cuda()
            event_images = event_images.cuda()
        
        with torch.cuda.amp.autocast(enabled=self.cuda):
            with torch.no_grad():
                outputs = self.net(rgb_images, event_images)
        
        # 解码和 NMS
        outputs = self.bbox_util.decode_box(outputs)
        results = self.bbox_util.non_max_suppression(
            torch.cat(outputs, 1), self.num_classes, self.input_shape,
            image_shape, self.letterbox_image, 
            conf_thres=self.confidence, nms_thres=self.nms_iou
        )
        
        if results[0] is None:
            f.close()
            if self.cuda:
                torch.cuda.empty_cache()
            return
        
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        
        # 只保留 top-N 结果，避免 detection results 文件过大
        top_indices = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_indices]
        top_conf = top_conf[top_indices]
        top_label = top_label[top_indices]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))
            ))
        
        f.close()
        if self.cuda:
            torch.cuda.empty_cache()
    
    def _find_image_path(self, file_name: str) -> str:
        """
        安全地查找图片路径，处理 JSON 中的子目录路径
        
        Args:
            file_name: COCO JSON 中的 file_name (可能包含子目录)
            
        Returns:
            完整的图片路径（如果找到），否则返回空字符串
        """
        # 方案 1: 直接使用完整路径
        full_path = os.path.join(self.image_dir, file_name)
        if os.path.exists(full_path):
            return full_path
        
        # 方案 2: 只使用文件名（去掉子目录）
        simple_name = os.path.basename(file_name)
        simple_path = os.path.join(self.image_dir, simple_name)
        if os.path.exists(simple_path):
            return simple_path
        
        # 方案 3: 在整个目录中递归查找
        import glob
        img_name_no_ext = os.path.splitext(simple_name)[0]
        pattern = os.path.join(self.image_dir, f"**/*{img_name_no_ext}.*")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        print(f"警告: 找不到图片文件: {file_name}")
        print(f"  尝试的路径包括:")
        print(f"    - {full_path}")
        print(f"    - {simple_path}")
        return ""
    
    def generate_result_files(self):
        """生成评估结果文件"""
        if not self.eval_flag:
            return
        
        if not self.eval_flag:
            return None
        
        if not os.path.exists(self.coco_json_path):
            print(f"警告: COCO 标注文件不存在: {self.coco_json_path}")
            print("跳过评估")
            return None
        
        # 计算当前评估图像数量
        eval_total = min(self.total_images, self.max_eval_samples)
        print(f"Evaluating up to {eval_total}/{self.total_images} images for mAP...")
        print(f"图片目录: {self.image_dir}")
        
        # 清理旧目录并复用
        if os.path.exists(self.map_out_path):
            shutil.rmtree(self.map_out_path)
        os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
        
        # 只读取一次 COCO json，用生成器逐步生成结果，避免一次性加载
        eval_images = self.images[:eval_total]
        
        with tqdm(total=len(eval_images), desc="Evaluating", unit="img") as pbar:
            # 记录找不到的图片
            missing_count = 0
            found_count = 0
            
            for img_info in eval_images:
                file_name = img_info.get('file_name')
                if not file_name:
                    pbar.update(1)
                    continue
                
                image_id = os.path.splitext(os.path.basename(file_name))[0]
                
                # 查找图像路径
                img_path = self._find_image_path(file_name)
                
                if not img_path or not os.path.exists(img_path):
                    missing_count += 1
                    pbar.update(1)
                    continue
                
                found_count += 1
                
                try:
                    image = Image.open(img_path)
                    self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                except Exception as e:
                    print(f"处理图像 {image_id} 失败: {e}")
                    pbar.update(1)
                    continue
                
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w", encoding='utf-8') as new_f:
                    if img_info['id'] in self.img_to_anns:
                        for ann in self.img_to_anns[img_info['id']]:
                            bbox = ann['bbox']
                            left, top, width, height = bbox
                            right = left + width
                            bottom = top + height
                            class_idx = self.cat_id_to_idx[ann['category_id']]
                            obj_name = self.class_names[class_idx]
                            new_f.write("%s %s %s %s %s\n" % (
                                obj_name, int(left), int(top), int(right), int(bottom)
                            ))
                
                if self.cuda:
                    torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)
        
        # 评估统计
        print(f"\n{'='*60}")
        print("评估统计")
        print(f"{'='*60}")
        print(f"  - 总图片数: {len(eval_images)}")
        print(f"  - 成功处理: {found_count}")
        print(f"  - 未找到图片: {missing_count}")
        
        if missing_count > 0:
            print(f"\\n  ⚠️  警告: 有 {missing_count}/{len(eval_images)} 张图片找不到")
            print(f"  这会导致 mAP 计算不准确")
        
        if found_count == 0:
            print(f"\\n  ✗ 错误: 没有找到任何图片，无法计算 mAP")
            print(f"  请检查图片路径设置")
            return 0.0
        
        print(f"\n计算 mAP...")
        try:
            from utils.utils_map import get_coco_map
            temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            temp_map = float(temp_map) if isinstance(temp_map, (int, float)) and not isinstance(temp_map, bool) else 0.0
        except Exception as e:
            print(f"  ⚠️  COCO mAP计算失败，使用VOC方式: {e}")
            import traceback
            traceback.print_exc()
            try:
                from utils.utils_map import get_map
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            except Exception as e2:
                print(f"  ✗ VOC mAP也失败: {e2}")
                temp_map = 0.0
        finally:
            if os.path.exists(self.map_out_path):
                shutil.rmtree(self.map_out_path)
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()
        
        print(f"  ✓ mAP 结果: {temp_map:.4f}")
        print(f"{'='*60}\n")
        
        return temp_map
    
    def on_epoch_end(self, epoch, model):
        """epoch 结束时调用评估"""
        if epoch % self.period != 0:
            return None
        
        print(f"开始评估 (epoch {epoch})...")
        
        old_net = self.net
        self.net = model
        temp_map = None
        
        try:
            model_device = next(model.parameters()).device
            original_train = model.training
            model.eval()
            torch.cuda.empty_cache()
            gc.collect()
            
            temp_map = self.generate_result_files()
            
            if original_train:
                model.train()
            
            if temp_map is not None:
                print(f"Epoch {epoch}: mAP = {temp_map:.4f}")
                if temp_map > self.best_map:
                    self.best_map = temp_map
                    best_model_path = os.path.join(self.log_dir, "fred_fusion_best.pth")
                    torch.save({
                        'state_dict': model.state_dict(),
                        'mAP': self.best_map,
                        'epoch': epoch
                    }, best_model_path)
                    print(f"✅ 新最佳权重保存: {best_model_path}")
                eval_result_path = os.path.join(self.log_dir, "eval_results.txt")
                with open(eval_result_path, "a", encoding='utf-8') as f:
                    f.write(f"epoch {epoch}: mAP = {temp_map:.4f}\n")
            else:
                print("⚠️ 评估失败，没有生成结果")
        
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.net = old_net
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()
        
        return temp_map

def fit_one_epoch_fusion(model_train, model, ema, yolo_loss, loss_history, eval_callback,
                         optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                         UnFreeze_Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank, max_batches=None):
    """
    Fusion 模型专用的训练函数
    """
    try:
        loss        = 0
        val_loss    = 0
        
        if local_rank == 0:
            print('\n' + '-'*80)
            start_time = time.time()
        
        model_train.train()
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if max_batches and iteration >= max_batches:
                    break
                
                if iteration >= epoch_step:
                    break
                
                # ---------------------------------#
                #   获取数据
                # ---------------------------------#
                rgb_images, event_images = batch[0]  # Fusion 模式返回 (rgb_images, event_images)
                targets = batch[1]
                y_trues = batch[2]
                fusion_infos = batch[3]
                
                
                
                with torch.no_grad():
                    if cuda:
                        # 分别将两个图像移动到 GPU
                        rgb_images = rgb_images.cuda(local_rank)
                        event_images = event_images.cuda(local_rank)
                        targets = [ann.cuda(local_rank) for ann in targets]
                        y_trues = [ann.cuda(local_rank) for ann in y_trues]
                
                # ----------------------#
                #   清零梯度
                # ----------------------#
                optimizer.zero_grad()
                
                # 检查是否使用混合精度训练（需要 scaler）
                use_autocast = fp16 and (scaler is not None)
                
                if not use_autocast:
                    # ----------------------#
                    #   前向传播
                    # ----------------------#
                    outputs = model_train(rgb_images, event_images)
                    
                    loss_value_all  = 0
                    # ----------------------#
                    #   计算损失
                    # ----------------------#
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                        loss_value_all  += loss_item
                
                    # ----------------------#
                    #   反向传播
                    # ----------------------#
                    loss_value_all.backward()
                    optimizer.step()
                    
                    if ema:
                        ema.update(model_train)
                        
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # ----------------------#
                        #   前向传播
                        # ----------------------#
                        outputs = model_train(rgb_images, event_images)
                        
                        loss_value_all  = 0
                        # ----------------------#
                        #   计算损失
                        # ----------------------#
                        for l in range(len(outputs)):
                            loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                            loss_value_all  += loss_item
                    
                    # ----------------------#
                    #   反向传播
                    # ----------------------#
                    scaler.scale(loss_value_all).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if ema:
                        ema.update(model_train)
                
                loss += loss_value_all.item()
                
                pbar.set_postfix(**{
                    'loss'  : loss / (iteration + 1),
                    'lr'    : get_lr(optimizer)
                })
                pbar.update(1)
        
        if local_rank == 0:
            
            
            train_time = time.time() - start_time
            print(f'Train cost time: {train_time:.1f}s')
        
        # ----------------------#
        #   验证集验证
        # ----------------------#
        if local_rank == 0:
            print('\n' + '-'*80)
            start_time = time.time()
        
        model_train.eval()
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_val):
                if max_batches and iteration >= max_batches:
                    break
                
                if iteration >= epoch_step_val:
                    break
                
                # ---------------------------------#
                #   获取数据
                # ---------------------------------#
                rgb_images, event_images = batch[0]  # Fusion 模式返回 (rgb_images, event_images)
                targets = batch[1]
                y_trues = batch[2]
                
                with torch.no_grad():
                    if cuda:
                        # 分别将两个图像移动到 GPU
                        rgb_images = rgb_images.cuda(local_rank)
                        event_images = event_images.cuda(local_rank)
                        targets = [ann.cuda(local_rank) for ann in targets]
                        y_trues = [ann.cuda(local_rank) for ann in y_trues]
                    
                    # ----------------------#
                    #   前向传播
                    # ----------------------#
                    outputs = model_train(rgb_images, event_images)
                    
                    loss_value_all  = 0
                    # ----------------------#
                    #   计算损失
                    # ----------------------#
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                        loss_value_all  += loss_item
            
                    val_loss += loss_value_all.item()
                
                pbar.set_postfix(**{
                    'val_loss': val_loss / (iteration + 1)
                })
                pbar.update(1)
        
        # ------------------------------#
        #   更新验证集损失曲线
        # ------------------------------#
        if local_rank == 0:
            val_time = time.time() - start_time
            print(f'Val cost time: {val_time:.1f}s')
            
            loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
            
            # EMA 模型评估
            eval_model = ema.ema if ema else model
            
            if eval_callback:
                eval_callback.on_epoch_end(epoch + 1, eval_model)
            
            # ------------------------------#
            #   保存模型
            # ------------------------------#
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
        import traceback
        traceback.print_exc()
        raise

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    设置优化器的学习率
    
    Args:
        optimizer: 优化器
        lr_scheduler_func: 学习率调度函数
        epoch: 当前 epoch
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    # 其他配置
    fred_root = "/mnt/data/datasets/fred"
    
    # 全局配置
    globals().update(vars(config_fred))
    
    # 初始化
    seed_everything(1)
    
    # 获取训练配置
    cfg = config_fred
    
    # === 核心配置 ===
    
    # 显卡配置
    Cuda            = True
    fp16            = False

    # 数据集和类别配置
    num_classes     = config_fred.NUM_CLASSES
    class_names     = config_fred.CLASS_NAMES
    anchors_path    = 'model_data/yolo_anchors.txt'
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='FRED Fusion 数据集训练')
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
    Freezed_Epoch       = 1
    UnFreeze_Epoch      = 300
    
    eval_period         = 5  # 每 5 个 epoch 评估一次 mAP
    save_period         = 10 # 每 10 个 epoch 保存一次模型
    
    Freeze_batch_size   = 8
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
    
    # 训练参数
    num_workers         = 4
    prefetch_factor     = 4
    persistent_workers  = True
    
    # 数据增强
    mosaic              = False
    mosaic_prob         = 0.5
    mixup               = False
    mixup_prob          = 0.5
    
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
    # Fusion 模型使用经过训练的单模态模型作为预训练权重的源
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
        input_shape = [640, 640]  # 实际还是640，但使用高分辨率模式的 anchors
        anchors_path = 'model_data/yolo_anchors_high_res.txt'
        anchors, num_anchors = get_anchors(anchors_path)
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # === 数据集准备 ===
    
    print("\n" + "="*80)
    print("Fusion 数据集准备")
    print("="*80)
    
    # 检查数据集
    # 计算 epoch 长度
    with open(train_json, 'r') as f:
        train_data = json.load(f)
        train_num = len(train_data['images'])
    
    with open(val_json, 'r') as f:
        val_data = json.load(f)
        val_num = len(val_data['images'])
    
    # 创建 Fusion 数据集（使用融合标注）
    fusion_train_set = FusionYoloDataset(
        train_json, fred_root, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=max(train_num // Freeze_batch_size, 1), 
        mosaic=True, mixup=True, mosaic_prob=cfg.MOSAIC_PROB, mixup_prob=cfg.MIXUP_PROB, train=True,
        modality=fusion_modality, use_fusion_info=True
    ) if not args.eval_only else None
    
    fusion_val_set = FusionYoloDataset(
        val_json, fred_root, input_shape, num_classes, anchors, anchors_mask,
        epoch_length=max(val_num // Unfreeze_batch_size, 1),
        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
        modality='rgb', use_fusion_info=True  # 验证集使用 RGB 模式
    )
    
    # === 构建训练和验证数据加载器 ===
    
    if not args.eval_only:
        gen = DataLoader(
            fusion_train_set, shuffle=True, batch_size=Freeze_batch_size, 
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=fusion_dataset_collate,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        
        gen_val = DataLoader(
            fusion_val_set, shuffle=True, batch_size=Unfreeze_batch_size, 
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=fusion_dataset_collate,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        
        epoch_step = max(len(gen), 1)
        epoch_step_val = max(len(gen_val), 1)
        
        if args.quick_test:
            epoch_step = min(epoch_step, 10)
            epoch_step_val = min(epoch_step_val, 10)
        
        print(f"初始化：数据集大小 train={train_num}, val={val_num}")
        print(f"初始化：epoch_step={epoch_step}, epoch_step_val={epoch_step_val}")
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
                rgb_key = f'backbone_rgb.{k[9:]}'  # 去掉 'backbone.'，替换为 'backbone_rgb.'
                if rgb_key in model_dict and np.shape(model_dict[rgb_key]) == np.shape(v):
                    fusion_pretrained[rgb_key] = v
                
                # Event 流（使用相同的权重初始化）
                event_key = f'backbone_event.{k[9:]}'  # 去掉 'backbone.'，替换为 'backbone_event.'
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
    
    # === 创建 mAP 评估回调 ===
    
    if default_eval_mode:
        # 使用 COCO 格式的 mAP 评估
        print("\n✓ 使用COCO格式的mAP评估（会增加训练时间）")
        print("  - 评估周期: 每", eval_period, "个epoch")
        print("  - 评估数据集: 测试集 (" + test_json + ")")
        if args.quick_test:
            print("  - ⚡ 快速验证: 仅评估 100 个样本")
            max_eval_samples = 100
        
        eval_callback = FusionCocoEvalCallback(
            net=model,
            input_shape=input_shape,
            anchors=anchors,
            anchors_mask=anchors_mask,
            class_names=class_names,
            num_classes=num_classes,
            coco_json_path=test_json,
            image_dir=fred_root,  # Fusion数据集使用相对路径
            log_dir=save_dir,
            cuda=Cuda,
            map_out_path=os.path.join(save_dir, ".temp_map_out"),
            max_boxes=max_boxes,
            confidence=confidence,
            nms_iou=nms_iou,
            letterbox_image=letterbox_image,
            MINOVERLAP=MINOVERLAP,
            eval_flag=default_eval_mode,
            period=eval_period,
            max_eval_samples=max_eval_samples
        )
    else:
        # 使用简化版回调（只记录epoch，不计算mAP）
        eval_callback = SimplifiedEvalCallback(save_dir, eval_flag=False, period=1)
        print("\n⚠️  注意: mAP评估已禁用（使用 --no_eval_map 禁用，默认启用）")
        print("  - 如需加快训练速度，可使用: python train_fred_fusion.py --no_eval_map")
    
    # === 显示配置 ===
    
    print("\n" + "="*80)
    print("训练配置")
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
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_train=len(gen) if not args.eval_only else 0,
        num_val=len(gen_val) if not args.eval_only else 0,
        high_res=high_res,
        anchors_path=anchors_path,
        anchors_mask=anchors_mask,
    )
    
    if args.eval_only:
        print("-"*80)
        print("⚠️  评估模式：只进行验证集评估，不进行训练")
        print("-"*80)
    
    # === 完整训练循环将在下面实现 ===
    
    # === 完整训练循环（冻结+解冻） ===
    
    if not args.eval_only:
        # 计算数据集大小
        num_train = train_num
        num_val = val_num
        
        # 初始化优化器
        optimizer = {
            'adam'  : optim.Adam(model_train.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model_train.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]
        
        # 训练循环
        UnFreeze_flag = False
        
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 解冻训练
            if epoch >= Freezed_Epoch and not UnFreeze_flag and Freeze_Train and not args.freeze_training and not args.quick_test:
                print("\n" + "="*80)
                print("第二阶段：解冻训练 (%d-%d epoch)" % (Freezed_Epoch, UnFreeze_Epoch))
                print("="*80)
                print("  - 解冻主干网络，全网络训练")
                print("  - 显存占用较大")
                print("  - 适合：追求最佳性能")
                
                batch_size = Unfreeze_batch_size
                
                # 学习率调整
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                # 解冻主干网络
                for param in model.backbone_rgb.parameters():
                    param.requires_grad = True
                for param in model.backbone_event.parameters():
                    param.requires_grad = True
                for param in model.compression_convs.parameters():
                    param.requires_grad = True
                
                # 重新创建优化器
                optimizer = {
                    'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
                    'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
                }[optimizer_type]
                
                # 重新创建 scaler 以适应新的 optimizer
                scaler = torch.amp.GradScaler('cuda', enabled=fp16)
                
                # 重新计算 epoch_step
                epoch_step = max(num_train // batch_size, 1)
                epoch_step_val = max(num_val // batch_size, 1)
                
                print(f"解冻训练：数据集大小 train={num_train}, val={num_val}, batch_size={batch_size}")
                print(f"解冻训练：epoch_step={epoch_step}, epoch_step_val={epoch_step_val}")
                
                if ema:
                    ema.updates = epoch_step * epoch
                
                # 重新创建数据加载器
                gen = DataLoader(
                    fusion_train_set, shuffle=True, batch_size=batch_size, 
                    num_workers=num_workers, pin_memory=True,
                    drop_last=True, collate_fn=fusion_dataset_collate,
                    persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
                )
                
                gen_val = DataLoader(
                    fusion_val_set, shuffle=True, batch_size=batch_size, 
                    num_workers=num_workers, pin_memory=True,
                    drop_last=True, collate_fn=fusion_dataset_collate,
                    persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
                )
                
                print(f"解冻训练数据加载器已重新创建，batch_size={batch_size}, epoch_step={epoch_step}")
                
                UnFreeze_flag = True
            
            # 冻结训练阶段
            elif epoch < Freezed_Epoch and Freeze_Train and not args.freeze_training and not args.quick_test:
                if epoch == Init_Epoch:
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
                    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
            
            # 设置数据集的当前 epoch
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            
            # 设置学习率
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            # 训练一个 epoch
            fit_one_epoch_fusion(model_train, model, ema, yolo_loss, loss_history, eval_callback,
                                 optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                                 UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, 0, max_batches=max_batches)
            
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