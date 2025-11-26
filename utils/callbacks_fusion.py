"""
Fusion 模型专用的评估回调
支持双模态输入（RGB + Event）进行 mAP 评估
"""
import os
import json
import shutil
import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .callbacks_coco import CocoEvalCallback
from .utils_bbox import DecodeBox
from .utils import cvtColor, preprocess_input, resize_image


class FusionCocoEvalCallback(CocoEvalCallback):
    """
    Fusion 模型专用的 COCO 评估回调
    支持多种评估模式：
    - rgb_only: 只使用 RGB 模态评估
    - event_only: 只使用 Event 模态评估
    - dual_avg: 使用双模态平均值评估
    """
    
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, 
                 coco_json_path, image_dir_rgb, image_dir_event, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, 
                 nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, 
                 eval_flag=True, period=1, max_eval_samples=None,
                 fusion_mode="rgb_only"):
        """
        Fusion 模型评估回调初始化
        
        Args:
            net: Fusion 模型
            input_shape: 输入尺寸 [H, W]
            anchors: 先验框
            anchors_mask: 先验框 mask
            class_names: 类别名称列表
            num_classes: 类别数量
            coco_json_path: COCO 标注 JSON 文件路径
            image_dir_rgb: RGB 图片目录
            image_dir_event: Event 图片目录
            log_dir: 日志目录
            cuda: 是否使用 CUDA
            map_out_path: mAP 计算临时目录
            max_boxes: 最大检测框数量
            confidence: 置信度阈值
            nms_iou: NMS 的 IOU 阈值
            letterbox_image: 是否使用 letterbox
            MINOVERLAP: mAP 计算的 IOU 阈值
            eval_flag: 是否进行评估
            period: 评估周期（每多少个 epoch 评估一次）
            max_eval_samples: 最大评估样本数（用于快速验证）
            fusion_mode: Fusion 评估模式
                - "rgb_only": 只使用 RGB 模态（默认，速度最快）
                - "event_only": 只使用 Event 模态
                - "dual_avg": 使用双模态平均值（需要支持）
                - "dual_concat": 拼接双模态（需要支持）
        """
        # 调用父类初始化（需要修改传参以匹配父类）
        super().__init__(
            net=net,
            input_shape=input_shape,
            anchors=anchors,
            anchors_mask=anchors_mask,
            class_names=class_names,
            num_classes=num_classes,
            coco_json_path=coco_json_path,
            image_dir=image_dir_rgb,  # 父类只需要一个 image_dir
            log_dir=log_dir,
            cuda=cuda,
            map_out_path=map_out_path,
            max_boxes=max_boxes,
            confidence=confidence,
            nms_iou=nms_iou,
            letterbox_image=letterbox_image,
            MINOVERLAP=MINOVERLAP,
            eval_flag=eval_flag,
            period=period,
            max_eval_samples=max_eval_samples
        )
        
        # Fusion 特有参数
        self.image_dir_rgb = image_dir_rgb
        self.image_dir_event = image_dir_event
        self.fusion_mode = fusion_mode
        
        # 重构 bbox_util 以匹配 Fusion 模型的调用方式
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, 
                                   (self.input_shape[0], self.input_shape[1]), 
                                   self.anchors_mask)
        
        print(f"✓ FusionCocoEvalCallback 初始化完成")
        print(f"  - RGB 图片目录: {self.image_dir_rgb}")
        print(f"  - Event 图片目录: {self.image_dir_event}")
        print(f"  - Fusion 模式: {self.fusion_mode}")
    
    def _prepare_fusion_inputs(self, image_data: np.ndarray, image_id: str):
        """
        准备 Fusion 模型的双模态输入
        
        Args:
            image_data: RGB 图片数据（经过预处理）
            image_id: 图片 ID，用于组装 Event 图片路径
            
        Returns:
            rgb_images: RGB 模态 tensor [B, C, H, W]
            event_images: Event 模态 tensor [B, C, H, W]
        """
        model_device = next(self.net.parameters()).device
        
        if self.fusion_mode == "rgb_only":
            # 只使用 RGB 模态，Event 模态用 RGB 填充
            rgb_tensor = torch.from_numpy(image_data).to(model_device, non_blocking=True)
            event_tensor = rgb_tensor.clone()
            return rgb_tensor, event_tensor
        
        elif self.fusion_mode == "event_only":
            # 只使用 Event 模态，RGB 模态用 Event 填充
            # 需要从 Event 图片目录加载图片
            event_path = os.path.join(self.image_dir_event, f"{image_id}.png")
            if not os.path.exists(event_path):
                # 如果找不到 Event 图片，回退到 RGB
                event_path = os.path.join(self.image_dir_event, f"{image_id}.jpg")
            
            if os.path.exists(event_path):
                event_image = Image.open(event_path)
                event_data = self._preprocess_image(event_image)
                event_tensor = torch.from_numpy(event_data).to(model_device, non_blocking=True)
            else:
                # 找不到 Event 图片，用 RGB 填充
                event_tensor = torch.from_numpy(image_data).to(model_device, non_blocking=True)
            
            rgb_tensor = event_tensor.clone()
            return rgb_tensor, event_tensor
        
        elif self.fusion_mode == "dual_avg":
            # 使用双模态平均值（简化版）
            rgb_tensor = torch.from_numpy(image_data).to(model_device, non_blocking=True)
            
            # 尝试加载 Event 图片
            event_path = os.path.join(self.image_dir_event, f"{image_id}.png")
            if not os.path.exists(event_path):
                event_path = os.path.join(self.image_dir_event, f"{image_id}.jpg")
            
            if os.path.exists(event_path):
                event_image = Image.open(event_path)
                event_data = self._preprocess_image(event_image)
                event_tensor = torch.from_numpy(event_data).to(model_device, non_blocking=True)
            else:
                event_tensor = rgb_tensor.clone()
            
            return rgb_tensor, event_tensor
        
        elif self.fusion_mode == "dual_concat":
            # 双模态拼接（需要 Fusion 模型支持）
            rgb_tensor = torch.from_numpy(image_data).to(model_device, non_blocking=True)
            
            # 尝试加载 Event 图片
            event_path = os.path.join(self.image_dir_event, f"{image_id}.png")
            if not os.path.exists(event_path):
                event_path = os.path.join(self.image_dir_event, f"{image_id}.jpg")
            
            if os.path.exists(event_path):
                event_image = Image.open(event_path)
                event_data = self._preprocess_image(event_image)
                event_tensor = torch.from_numpy(event_data).to(model_device, non_blocking=True)
            else:
                event_tensor = rgb_tensor.clone()
            
            return rgb_tensor, event_tensor
        
        else:
            # 默认行为：RGB only
            rgb_tensor = torch.from_numpy(image_data).to(model_device, non_blocking=True)
            event_tensor = rgb_tensor.clone()
            return rgb_tensor, event_tensor
    
    def _preprocess_image(self, image):
        """预处理单张图片"""
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), 
                                 self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        return image_data
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """
        生成预测结果 txt 文件（支持 Fusion 模型）
        
        Args:
            image_id: 图片 ID
            image: PIL Image 对象（RGB）
            class_names: 类别名称列表
            map_out_path: 输出路径
        """
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), 
                "w", encoding='utf-8')
        
        try:
            # 预处理 RGB 图片
            image_shape = np.array(np.shape(image)[0:2])
            image_data = self._preprocess_image(image)
            
            # 准备 Fusion 模型输入
            rgb_images, event_images = self._prepare_fusion_inputs(image_data, image_id)
            
            # 使用混合精度推理（如果 CUDA 可用）
            with torch.cuda.amp.autocast(enabled=self.cuda and rgb_images.device.type == 'cuda'):
                with torch.no_grad():
                    # Fusion 模型前向传播
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
            top_confidence = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            
            # 只保留 top-N 个结果
            top_indices = np.argsort(top_confidence)[::-1][:self.max_boxes]
            top_boxes = top_boxes[top_indices]
            top_confidence = top_confidence[top_indices]
            top_label = top_label[top_indices]
            
            for i, c in list(enumerate(top_label)):
                predicted_class = class_names[int(c)]
                box = top_boxes[i]
                score = str(top_confidence[i])
                
                top, left, bottom, right = box
                if predicted_class not in class_names:
                    continue
                
                f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], 
                    str(int(left)), str(int(top)), 
                    str(int(right)), str(int(bottom))
                ))
            
            f.close()
            
            # 清理显存
            if self.cuda:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️  - 评估图片 {image_id} 失败: {e}")
            f.close()
            if self.cuda:
                torch.cuda.empty_cache()
    
    def on_epoch_end(self, epoch, model_eval):
        """Epoch 结束时的回调"""
        if epoch % self.period != 0 or not self.eval_flag:
            return
        
        self.net = model_eval
        self.net.eval()
        
        # 创建输出目录
        os.makedirs(self.map_out_path, exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Fusion 模型评估 (Epoch {epoch})")
        print(f"评测模式: {self.fusion_mode}")
        print(f"{'='*60}")
        
        # 快速验证模式
        eval_images = self.images
        if self.max_eval_samples is not None:
            eval_images = self.images[:self.max_eval_samples]
            print(f"⚡ 快速验证模式: 仅评估 {len(eval_images)} 个样本（共 {len(self.images)} 个）")
        
        # 遍历验证集
        for img_info in tqdm(eval_images, desc="评估", unit="img"):
            file_name = img_info.get('file_name') or img_info.get('rgb_file_name')
            if not file_name:
                continue
            
            # 提取图片 ID
            image_id = os.path.splitext(os.path.basename(file_name))[0]
            img_path = os.path.join(self.image_dir_rgb, file_name)
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # 读取 RGB 图片
                image = Image.open(img_path)
                
                # 生成预测结果
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                # 生成真实框 txt
                gt_path = os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt")
                with open(gt_path, "w", encoding='utf-8') as gt_f:
                    if img_info['id'] in self.img_to_anns:
                        for ann in self.img_to_anns[img_info['id']]:
                            bbox = ann['bbox']  # [x, y, width, height]
                            
                            # 转换为 [left, top, right, bottom]
                            left = int(bbox[0])
                            top = int(bbox[1])
                            right = int(bbox[0] + bbox[2])
                            bottom = int(bbox[1] + bbox[3])
                            
                            # 获取类别
                            class_idx = self.cat_id_to_idx[ann['category_id']]
                            obj_name = self.class_names[class_idx]
                            
                            gt_f.write("%s %s %s %s %s\n" % (
                                obj_name, left, top, right, bottom
                            ))
            
            except Exception as e:
                print(f"⚠️  - 处理图片 {image_id} 失败: {e}")
                continue
        
        # 计算 mAP
        print("计算 mAP...")
        try:
            from .utils_map import get_coco_map
            temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            temp_map = float(temp_map) if isinstance(temp_map, (int, float)) else 0.0
        except Exception as e:
            print(f"⚠️  - COCO mAP 计算失败，使用 VOC 方式: {e}")
            from .utils_map import get_map
            temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
        
        self.maps.append(temp_map)
        self.epoches.append(epoch)
        
        # 保存 mAP 记录
        map_file = os.path.join(self.log_dir, "epoch_map.txt")
        with open(map_file, 'a') as f:
            f.write(str(temp_map))
            f.write("\n")
        
        # 绘制 mAP 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='Validation mAP')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'mAP@{self.MINOVERLAP}', fontsize=12)
        plt.title('Fusion Model mAP Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.ylim(0, 1)
        
        plt.savefig(os.path.join(self.log_dir, "epoch_map.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 计算最佳 mAP
        if temp_map > max(self.maps[:-1], default=0):
            print(f"🎉  新最佳 mAP: {temp_map:.4f}")
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  - mAP@{self.MINOVERLAP}: {temp_map:.4f}")
        print(f"  - 当前最佳: {max(self.maps):.4f}")
        print(f"  - 评估样本: {len(eval_images)}/{len(self.images)}")
        print(f"  - Fusion 模式: {self.fusion_mode}")
        print(f"{'='*60}\n")
        
        # 清理临时文件
        try:
            shutil.rmtree(self.map_out_path, ignore_errors=True)
        except:
            pass


class FusionSimplifiedEvalCallback:
    """
    简化版 Fusion 评估回调
    只计算验证集 loss，不计算 mAP（适合快速训练）
    """
    def __init__(self, log_dir, eval_flag=True, period=1):
        """
        Args:
            log_dir: 日志目录
            eval_flag: 是否进行评估
            period: 评估周期
        """
        self.log_dir = log_dir
        self.eval_flag = eval_flag
        self.period = period
        
        self.epoches = [0]
        
        if self.eval_flag:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, model_eval):
        """Epoch 结束时的回调"""
        if epoch % self.period == 0 and self.eval_flag:
            self.epoches.append(epoch)
            print(f"✓ Epoch {epoch} 完成 (简化评估)")