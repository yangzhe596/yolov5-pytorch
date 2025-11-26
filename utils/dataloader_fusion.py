"""
FRED Fusion 数据集专用数据加载器
支持双模态（RGB + Event）配对训练
基于 COCO 格式的融合标注
"""
import json
import os
from random import sample, shuffle

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


class FusionYoloDataset(Dataset):
    """FRED Fusion 数据集专用加载器
    
    特性:
    - 支持双模态（RGB + Event）配对读取
    - 支持融合标注过滤（可选）
    - 优化的 Mosaic 数据增强（双模态同步增强）
    - 支持模态选择（RGB, Event, Dual）
    """
    
    def __init__(self, coco_json_path, fred_root, input_shape, num_classes, anchors, anchors_mask, 
                 epoch_length, mosaic, mixup, mosaic_prob, mixup_prob, train, 
                 special_aug_ratio=0.7, high_res=False, four_features=False,
                 modality='dual', use_fusion_info=True):
        """
        初始化 Fusion 数据集加载器
        
        Args:
            coco_json_path: COCO 格式的融合标注文件路径
            fred_root: FRED 数据集根目录（如 /mnt/data/datasets/fred）
            input_shape: 输入图像尺寸 [height, width]
            num_classes: 类别数量
            anchors: 先验框坐标
            anchors_mask: 先验框掩码
            epoch_length: 每个 epoch 的长度（用于动态调整数据增强比例）
            mosaic: 是否启用 Mosaic 数据增强
            mixup: 是否启用 MixUp 数据增强
            mosaic_prob: Mosaic 概率
            mixup_prob: MixUp 概率
            train: 是否为训练模式
            special_aug_ratio: 特殊数据增强比例（默认前 70% 轮次使用强增强）
            high_res: 是否启用高分辨率模式
            four_features: 是否启用四特征层模式
            modality: 模态选择 'dual'（默认）, 'rgb', 'event'
            use_fusion_info: 是否使用融合信息（时间戳、配对状态等）
        """
        super(FusionYoloDataset, self).__init__()
        
        self.coco_json_path = coco_json_path
        self.fred_root = fred_root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.high_res = high_res
        self.four_features = four_features
        self.modality = modality
        self.use_fusion_info = use_fusion_info
        
        # 验证模态参数
        if self.modality not in ['dual', 'rgb', 'event']:
            raise ValueError(f"不支持的模态: {self.modality}，必须是 'dual', 'rgb', 'event' 之一")
        
        self.epoch_now = -1
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4
        
        # 根据高分辨率模式设置步长
        if high_res:
            if len(anchors_mask) == 4:
                self.strides = {0: 32, 1: 16, 2: 8, 3: 4}  # 20x20, 40x40, 80x80, 160x160
            else:
                self.strides = {0: 16, 1: 8, 2: 4}  # 40x40, 80x80, 160x160
        else:
            self.strides = {0: 32, 1: 16, 2: 8}  # 40x40, 80x80, 20x20
        
        # 加载融合标注
        self._load_fusion_annotations()
        
        self.length = len(self.image_infos)
        
        print(f"\nFusion 数据集配置:")
        print(f"  - 标注文件: {self.coco_json_path}")
        print(f"  - FRED 根目录: {self.fred_root}")
        print(f"  - 输入尺寸: {self.input_shape}")
        print(f"  - 模态: {self.modality}")
        print(f"  - 总图片数: {self.length}")
        print(f"  - 使用融合信息: {self.use_fusion_info}")
    
    def _load_fusion_annotations(self):
        """加载 FRED Fusion 格式的 COCO 标注"""
        print(f"加载融合标注: {self.coco_json_path}")
        
        with open(self.coco_json_path, 'r') as f:
            fusion_data = json.load(f)
        
        # 创建 category_id 到索引的映射
        self.cat_id_to_idx = {}
        for cat in fusion_data['categories']:
            self.cat_id_to_idx[cat['id']] = cat['id'] - 1
        
        # 创建 image_id 到 annotations 的映射
        img_to_anns = {}
        for ann in fusion_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # 构建图片信息列表
        self.image_infos = []
        total_rgb = 0
        total_event = 0
        total_dual = 0
        
        for img in fusion_data['images']:
            img_id = img['id']
            
            # 根据模态过滤图片
            img_modality = img.get('modality', 'dual')
            if self.modality == 'dual' and img_modality != 'dual':
                continue
            elif self.modality == 'rgb' and img_modality not in ['dual', 'rgb']:
                continue
            elif self.modality == 'event' and img_modality not in ['dual', 'event']:
                continue
            
            if img_id in img_to_anns:
                # 构建 RGB 图像路径
                rgb_path = os.path.join(self.fred_root, img['rgb_file_name'])
                
                # 构建 Event 图像路径
                event_path = os.path.join(self.fred_root, img['event_file_name'])
                
                # 转换 COCO bbox 为 VOC 格式 [xmin, ymin, xmax, ymax, class_id]
                boxes = []
                for ann in img_to_anns[img_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[0] + bbox[2]
                    ymax = bbox[1] + bbox[3]
                    class_idx = self.cat_id_to_idx[ann['category_id']]
                    
                    boxes.append([xmin, ymin, xmax, ymax, class_idx])
                
                info = {
                    'rgb_path': rgb_path,
                    'event_path': event_path,
                    'boxes': np.array(boxes, dtype=np.float32),
                    'width': img['width'],
                    'height': img['height'],
                    'modality': img_modality
                }
                
                # 添加融合信息
                if self.use_fusion_info:
                    info['rgb_timestamp'] = img.get('rgb_timestamp', 0)
                    info['event_timestamp'] = img.get('event_timestamp', 0)
                    info['time_diff'] = img.get('time_diff', 0)
                    info['fusion_status'] = img.get('fusion_status', 'unknown')
                    info['sequence_id'] = img.get('sequence_id', 0)
                
                self.image_infos.append(info)
                
                # 统计模态
                if img_modality == 'dual':
                    total_dual += 1
                elif img_modality == 'rgb':
                    total_rgb += 1
                elif img_modality == 'event':
                    total_event += 1
        
        print(f"✓ 加载完成")
        print(f"  - 双模态(Dual): {total_dual}")
        print(f"  - 仅 RGB: {total_rgb}")
        print(f"  - 仅 Event: {total_event}")
        print(f"  - 符合当前模态的图片: {len(self.image_infos)}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """获取训练样本
        
        Returns:
            image: 预处理后的图像 [C, H, W]
            boxes: 边界框信息 [N, 5] or []
            y_true: 训练目标 (多层特征图)
            fusion_info: 融合信息（字典）或 None
        """
        index = index % self.length
        
        # 根据配置选择模态
        if self.modality == 'rgb':
            image, box = self._load_rgb_data(index)
        elif self.modality == 'event':
            image, box = self._load_event_data(index)
        else:  # dual
            # Dual模式：同时加载 RGB 和 Event
            rgb_image, rgb_box = self._load_rgb_data(index)
            event_image, event_box = self._load_event_data(index)
        
        # 数据增强
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            if self.modality == 'rgb':
                # RGB Mosaic
                indices = sample(range(self.length), 3)
                indices.append(index)
                shuffle(indices)
                image, box = self.get_random_data_with_Mosaic_RGB(indices, self.input_shape)
            elif self.modality == 'event':
                # Event Mosaic
                indices = sample(range(self.length), 3)
                indices.append(index)
                shuffle(indices)
                image, box = self.get_random_data_with_Mosaic_Event(indices, self.input_shape)
            else:
                # Dual Mosaic（同时增强 RGB 和 Event）
                indices = sample(range(self.length), 3)
                indices.append(index)
                shuffle(indices)
                rgb_image, event_image, box = self.get_random_dual_mosaic(indices, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                mix_index = sample(range(self.length), 1)[0]
                if self.modality == 'rgb':
                    image_2, box_2 = self._load_rgb_data(mix_index, augment=True)
                elif self.modality == 'event':
                    image_2, box_2 = self._load_event_data(mix_index, augment=True)
                else:
                    # Dual 模式的 MixUp
                    rgb_image_2, event_image_2, box_2 = self.get_random_dual_mosaic([mix_index, mix_index, mix_index, mix_index], self.input_shape)
                    rgb_image, rgb_box = self.get_random_data_with_MixUp(rgb_image, box, rgb_image_2, box_2)
                    event_image, box = self.get_random_data_with_MixUp(event_image, box, event_image_2, box_2)
                if self.modality != 'dual':
                    image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            # 不使用 Mosaic，使用单样本增强
            if self.modality == 'rgb':
                image, box = self._load_rgb_data(index, augment=True)
            elif self.modality == 'event':
                image, box = self._load_event_data(index, augment=True)
            else:
                # Dual 模式：同时加载 RGB 和 Event
                rgb_image, rgb_box = self._load_rgb_data(index, augment=True)
                event_image, event_box = self._load_event_data(index, augment=True)
                # 对于 Dual 模式，只使用 RGB 的边界框（假设对齐）
                box = rgb_box
        
        # 预处理
        if self.modality == 'dual':
            # Dual 模式：分别预处理两个图像
            rgb_image = np.transpose(preprocess_input(np.array(rgb_image, dtype=np.float32)), (2, 0, 1))
            event_image = np.transpose(preprocess_input(np.array(event_image, dtype=np.float32)), (2, 0, 1))
        
        if self.modality != 'dual':
            image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        
        box = np.array(box, dtype=np.float32)
        
        if len(box) != 0:
            # 归一化边界框
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # 转换为中心点和宽高格式
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        y_true = self.get_target(box)
        
        # 返回融合信息（可选）
        fusion_info = None
        if self.use_fusion_info:
            info = self.image_infos[index % len(self.image_infos)]
            fusion_info = {
                'modality': info['modality'],
                'rgb_timestamp': info.get('rgb_timestamp', 0),
                'event_timestamp': info.get('event_timestamp', 0),
                'time_diff': info.get('time_diff', 0),
                'fusion_status': info.get('fusion_status', 'unknown'),
                'sequence_id': info.get('sequence_id', 0)
            }
        
        if self.modality == 'dual':
            # Dual 模式返回两个图像
            return (rgb_image, event_image), box, y_true, fusion_info
        else:
            # 单模态返回单个图像
            return image, box, y_true, fusion_info
    
    def rand(self, a=0, b=1):
        """生成随机数"""
        return np.random.rand() * (b - a) + a
    
    def _load_image_cv2(self, image_path):
        """加载图像（使用 cv2）"""
        image = cv2.imread(image_path)
        if image is None:
            # 如果图片不存在，创建随机噪声图片（避免训练中断）
            print(f"警告: 图片不存在，使用随机噪声替代: {image_path}")
            image = np.random.randint(0, 255, (self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_rgb_data(self, index, augment=True):
        """加载 RGB 数据"""
        info = self.image_infos[index % len(self.image_infos)]
        image = self._load_image_cv2(info['rgb_path'])
        box = info['boxes'].copy()
        
        if augment:
            return self._apply_augmentation(image, box)
        else:
            return self._apply_letterbox(image, box)
    
    def _load_event_data(self, index, augment=True):
        """加载 Event 数据"""
        info = self.image_infos[index % len(self.image_infos)]
        image = self._load_image_cv2(info['event_path'])
        box = info['boxes'].copy()
        
        if augment:
            return self._apply_augmentation(image, box)
        else:
            return self._apply_letterbox(image, box)
    
    def _apply_letterbox(self, image, box):
        """Letterbox 缩放（保持长宽比，填充黑边）"""
        ih, iw = image.shape[:2]
        h, w = self.input_shape
        
        # 计算缩放比例
        scale = min(w/iw, h/ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        
        # 缩放并填充
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        new_image[dy:dy+nh, dx:dx+nw, :] = image
        
        # 调整边界框
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            
            # 过滤边界框
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        
        return new_image, box
    
    def _apply_augmentation(self, image, box):
        """应用数据增强"""
        ih, iw = image.shape[:2]
        h, w = self.input_shape
        jitter = 0.3
        hue = 0.1
        sat = 0.7
        val = 0.4
        
        # 随机缩放和扭曲
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # 放置到新图像（黑边填充）
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        
        # 安全复制（避免越界）
        if nw > w or nh > h:
            src_x1 = max(0, -dx)
            src_y1 = max(0, -dy)
            src_x2 = min(nw, w - dx)
            src_y2 = min(nh, h - dy)
            
            dst_x1 = max(0, dx)
            dst_y1 = max(0, dy)
            dst_x2 = min(w, dx + nw)
            dst_y2 = min(h, dy + nh)
            
            new_image[dst_y1:dst_y2, dst_x1:dst_x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
        else:
            new_image[dy:dy+nh, dx:dx+nw, :] = image
        
        # 水平翻转
        flip = self.rand() < 0.5
        if flip:
            new_image = cv2.flip(new_image, 1)
        
        # HSV 色域变换
        image_data = self._apply_hsv_augmentation(new_image, hue, sat, val)
        
        # 调整边界框
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        
        return image_data, box
    
    def _apply_hsv_augmentation(self, image, hue=0.1, sat=0.7, val=0.4):
        """HSV 色域变换"""
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        dtype = image.dtype
        x = np.arange(0, 256, dtype=np.float32)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        h = cv2.LUT(h, lut_hue)
        s = cv2.LUT(s, lut_sat)
        v = cv2.LUT(v, lut_val)
        
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return image
    
    def get_random_data_with_Mosaic_RGB(self, indices, input_shape):
        """RGB Mosaic 数据增强"""
        return self._get_random_mosaic(indices, input_shape, modality='rgb')
    
    def get_random_data_with_Mosaic_Event(self, indices, input_shape):
        """Event Mosaic 数据增强"""
        return self._get_random_mosaic(indices, input_shape, modality='event')
    
    def get_random_dual_mosaic(self, indices, input_shape):
        """双模态 Mosaic 数据增强（同时处理 RGB 和 Event）"""
        # 同时为 RGB 和 Event 创建 Mosaic
        rgb_image, rgb_box = self._get_random_mosaic(indices, input_shape, modality='rgb')
        event_image, event_box = self._get_random_mosaic(indices, input_shape, modality='event')
        # 对于 Dual 模式，使用 RGB 的边界框
        return rgb_image, event_image, rgb_box
    
    def _get_random_mosaic(self, indices, input_shape, modality='rgb'):
        """通用 Mosaic 实现"""
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        
        # 预分配图像（减少内存分配）
        mosaic_image = np.full((h, w, 3), 128, dtype=np.uint8)
        box_datas = []
        
        for idx, i in enumerate(indices):
            info = self.image_infos[i % len(self.image_infos)]
            
            # 根据 modality 选择图像
            if modality == 'rgb':
                image = self._load_image_cv2(info['rgb_path'])
            else:
                image = self._load_image_cv2(info['event_path'])
            
            ih, iw = image.shape[:2]
            box = info['boxes'].copy()
            
            # 翻转
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = cv2.flip(image, 1)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            
            # 缩放
            new_ar = iw/ih * self.rand(0.8, 1.2) / self.rand(0.8, 1.2)
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # 计算放置位置
            if idx == 0:  # 左上
                dx = cutx - nw
                dy = cuty - nh
            elif idx == 1:  # 右上
                dx = cutx
                dy = cuty - nh
            elif idx == 2:  # 左下
                dx = cutx - nw
                dy = cuty
            else:  # 右下
                dx = cutx
                dy = cuty
            
            # 计算有效区域
            x1a = max(0, dx)
            y1a = max(0, dy)
            x2a = min(w, dx + nw)
            y2a = min(h, dy + nh)
            
            x1b = max(0, -dx)
            y1b = max(0, -dy)
            x2b = min(nw, w - dx)
            y2b = min(nh, h - dy)
            
            # 复制到 Mosaic 图像
            if x2a > x1a and y2a > y1a:
                mosaic_image[y1a:y2a, x1a:x2a, :] = image[y1b:y2b, x1b:x2b, :]
            
            # 调整边界框
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                
                if len(box) > 0:
                    box_data = np.zeros((len(box), 5))
                    box_data[:len(box)] = box
                    box_datas.append(box_data)
                else:
                    box_datas.append(np.array([]))
            else:
                box_datas.append(np.array([]))
        
        # 合并边界框
        new_boxes = np.array(self._merge_mosaic_bboxes(box_datas, cutx, cuty))
        return mosaic_image, new_boxes
    
    def _merge_mosaic_bboxes(self, bboxes, cutx, cuty):
        """合并 Mosaic 的边界框"""
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if i == 0:  # 左上
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                elif i == 1:  # 右上
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                elif i == 2:  # 左下
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                else:  # 右下
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                merge_bbox.append([x1, y1, x2, y2, box[-1]])
        
        return merge_bbox
    
    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        """MixUp 数据增强"""
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    def get_target(self, targets):
        """生成训练目标"""
        num_layers = len(self.anchors_mask)
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // self.strides[l] for l in range(num_layers)]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / self.strides[l]
            
            batch_target = np.zeros_like(targets)
            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h
            batch_target[:, 4] = targets[:, 4]
            
            valid_mask = (batch_target[:, 2] > 0) & (batch_target[:, 3] > 0)
            if not np.any(valid_mask):
                continue
            
            valid_batch_target = batch_target[valid_mask]
            
            epsilon = 1e-8
            ratios_of_gt_anchors = np.expand_dims(valid_batch_target[:, 2:4], 1) / (np.expand_dims(anchors, 0) + epsilon)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / (np.expand_dims(valid_batch_target[:, 2:4], 1) + epsilon)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            max_ratios = np.max(ratios, axis=-1)
            
            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    
                    i = int(np.floor(valid_batch_target[t, 0]))
                    j = int(np.floor(valid_batch_target[t, 1]))
                    
                    offsets = self._get_near_points(valid_batch_target[t, 0], valid_batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue
                        
                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                        
                        c = int(valid_batch_target[t, 4])
                        
                        y_true[l][k, local_j, local_i, 0] = valid_batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = valid_batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = valid_batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = valid_batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
        
        return y_true
    
    def _get_near_points(self, x, y, i, j):
        """获取附近的网格点"""
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]


def fusion_dataset_collate(batch):
    """Fusion 数据集的 collate 函数
    
    返回:
        images: 元组 (rgb_images, event_images) 或者 images
        boxes: 边界框
        y_trues: 训练目标
        fusion_infos: 融合信息
    """
    rgb_images = []
    event_images = []
    bboxes = []
    y_trues = [[] for _ in range(len(batch[0][2]))]
    fusion_infos = []
    
    # 检查是否是Dual模式（包含RGB和Event）
    is_dual = False
    if len(batch) > 0:
        first_item = batch[0]
        # 检查第一个元素是否是元组或列表（包含两个图像）
        if isinstance(first_item[0], (tuple, list)) and len(first_item[0]) == 2:
            is_dual = True
    
    if is_dual:
        # Dual模式：每个样本包含 (rgb_img, event_img, box, y_true, fusion_info)
        for imgs, box, y_true, fusion_info in batch:
            rgb_img, event_img = imgs
            rgb_images.append(rgb_img)
            event_images.append(event_img)
            bboxes.append(box)
            for i, y in enumerate(y_true):
                y_trues[i].append(y)
            if fusion_info is not None:
                fusion_infos.append(fusion_info)
        
        rgb_images = torch.from_numpy(np.array(rgb_images)).type(torch.FloatTensor)
        event_images = torch.from_numpy(np.array(event_images)).type(torch.FloatTensor)
    else:
        # 单模态模式：每个样本是 (img, box, y_true, fusion_info)
        for img, box, y_true, fusion_info in batch:
            rgb_images.append(img)
            bboxes.append(box)
            for i, y in enumerate(y_true):
                y_trues[i].append(y)
            if fusion_info is not None:
                fusion_infos.append(fusion_info)
        
        rgb_images = torch.from_numpy(np.array(rgb_images)).type(torch.FloatTensor)
        # 在单模态模式下，event_images 与 rgb_images 相同
        event_images = rgb_images
    
    bboxes = [torch.from_numpy(np.array(ann)).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann)).type(torch.FloatTensor) for ann in y_trues]
    
    return (rgb_images, event_images), bboxes, y_trues, fusion_infos if fusion_infos else None