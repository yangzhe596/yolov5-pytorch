"""
COCO格式数据加载器
支持FRED数据集的COCO格式
已优化Mosaic数据增强性能
"""
import json
import os
from random import sample, shuffle

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


class CocoYoloDataset(Dataset):
    """COCO格式的YOLO数据集加载器"""
    
    def __init__(self, coco_json_path, image_dir, input_shape, num_classes, anchors, anchors_mask, 
                 epoch_length, mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7, high_res=False, four_features=False):
        super(CocoYoloDataset, self).__init__()
        
        self.coco_json_path = coco_json_path
        self.image_dir = image_dir
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
        self.high_res = high_res  # 是否启用高分辨率模式
        self.four_features = four_features  # 是否启用四特征层模式
        
        self.epoch_now = -1
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4
        
        # 根据高分辨率模式设置步长
        if high_res:
            if len(anchors_mask) == 4:
                # 四特征层高分辨率模式：20x20, 40x40, 80x80, 160x160特征层（按照模型输出顺序）
                self.strides = {0: 32, 1: 16, 2: 8, 3: 4}  # 对应20x20, 40x40, 80x80, 160x160
            else:
                # 三特征层高分辨率模式：40x40, 80x80, 160x160特征层（按照模型输出顺序）
                self.strides = {0: 16, 1: 8, 2: 4}  # 对应40x40, 80x80, 160x160
        else:
            # 标准模式：40x40, 80x80, 20x20特征层（按照模型输出顺序）
            self.strides = {0: 32, 1: 16, 2: 8}  # 对应40x40, 80x80, 20x20
        
        # 加载COCO标注
        self._load_coco_annotations()
        
        self.length = len(self.image_infos)
    
    def _load_coco_annotations(self):
        """加载COCO格式的标注文件"""
        print(f"加载COCO标注: {self.coco_json_path}")
        
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 创建category_id到索引的映射（COCO的category_id从1开始）
        self.cat_id_to_idx = {}
        for cat in coco_data['categories']:
            # COCO category_id从1开始，转换为0开始的索引
            self.cat_id_to_idx[cat['id']] = cat['id'] - 1
        
        # 创建image_id到annotations的映射
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # 构建图片信息列表（只包含有标注的图片）
        self.image_infos = []
        for img in coco_data['images']:
            if img['id'] in img_to_anns:
                # 支持 Fusion 数据集的文件名字段
                # 优先使用 file_name，如果没有则使用 rgb_file_name 或 event_file_name
                file_name = img.get('file_name') or img.get('rgb_file_name') or img.get('event_file_name')
                if not file_name:
                    continue
                
                # 支持相对路径格式（如 "序列0/PADDED_RGB/xxx.jpg"）
                # image_dir 应该是 FRED 数据集的根目录
                img_path = os.path.join(self.image_dir, file_name)
                
                # 转换COCO bbox为VOC格式 [xmin, ymin, xmax, ymax, class_id]
                boxes = []
                for ann in img_to_anns[img['id']]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[0] + bbox[2]
                    ymax = bbox[1] + bbox[3]
                    class_idx = self.cat_id_to_idx[ann['category_id']]
                    
                    boxes.append([xmin, ymin, xmax, ymax, class_idx])
                
                self.image_infos.append({
                    'path': img_path,
                    'boxes': np.array(boxes, dtype=np.float32),
                    'width': img['width'],
                    'height': img['height']
                })
        
        print(f"✓ 加载完成: {len(self.image_infos)} 张图片")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index % self.length
        
        # 训练时进行数据增强
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            indices = sample(range(self.length), 3)
            indices.append(index)
            shuffle(indices)
            
            image, box = self.get_random_data_with_Mosaic(indices, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                mix_index = sample(range(self.length), 1)[0]
                image_2, box_2 = self.get_random_data(mix_index, self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(index, self.input_shape, random=self.train)
        
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        
        if len(box) != 0:
            # 对真实框进行归一化，调整到0-1之间
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # 转换为中心点和宽高格式
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        y_true = self.get_target(box)
        return image, box, y_true
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def _load_image_cv2(self, image_path):
        """使用cv2加载图像（比PIL快）"""
        image = cv2.imread(image_path)
        if image is None:
            # 如果图片不存在，创建一个随机噪声图片作为替代
            # 这样可以让训练继续进行，而不是因为缺失图片而中断
            print(f"警告: 图片不存在，使用随机噪声图片替代: {image_path}")
            # 创建一个随机噪声图片，大小与输入尺寸匹配，RGB格式
            # 使用self.input_shape[1]作为宽度，self.input_shape[0]作为高度
            image = np.random.randint(0, 255, (self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        # cv2读取的是BGR格式，需要转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _apply_hsv_augmentation(self, image, hue=0.1, sat=0.7, val=0.4):
        """优化的HSV色域变换"""
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
    
    def get_random_data(self, index, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """获取随机数据（单张图片）"""
        info = self.image_infos[index]
        
        # 使用cv2读取图像
        image = self._load_image_cv2(info['path'])
        ih, iw = image.shape[:2]
        h, w = input_shape
        
        # 获得预测框
        box = info['boxes'].copy()
        
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            
            # 将图像多余的部分加上灰条
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            
            # 调整边界框
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            
            return new_image, box
        
        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        
        # 计算有效的放置区域，避免索引越界
        if nw > w or nh > h:
            # 如果缩放后的图像大于目标尺寸，需要裁剪
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
        
        # 翻转图像
        flip = self.rand()<.5
        if flip: new_image = cv2.flip(new_image, 1)
        
        # 对图像进行色域变换
        image_data = self._apply_hsv_augmentation(new_image, hue, sat, val)
        
        # 调整边界框
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        
        return image_data, box
    
    def get_random_data_with_Mosaic(self, indices, input_shape):
        """优化的Mosaic数据增强"""
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        
        # 预分配Mosaic图像（避免多次内存分配）
        mosaic_image = np.full((h, w, 3), 128, dtype=np.uint8)
        box_datas = []
        
        for idx, i in enumerate(indices):
            info = self.image_infos[i]
            
            # 使用cv2读取图像
            image = self._load_image_cv2(info['path'])
            ih, iw = image.shape[:2]
            
            box = info['boxes'].copy()
            
            # 翻转图像
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = cv2.flip(image, 1)
                box[:, [0,2]] = iw - box[:, [2,0]]
            
            # 对图像进行缩放
            new_ar = iw/ih * self.rand(0.8, 1.2) / self.rand(0.8, 1.2)
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
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
            
            # 直接复制到Mosaic图像（避免创建临时图像）
            if x2a > x1a and y2a > y1a:
                mosaic_image[y1a:y2a, x1a:x2a, :] = image[y1b:y2b, x1b:x2b, :]
            
            # 调整边界框
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                
                if len(box) > 0:
                    box_data = np.zeros((len(box),5))
                    box_data[:len(box)] = box
                    box_datas.append(box_data)
                else:
                    box_datas.append(np.array([]))
            else:
                box_datas.append(np.array([]))
        
        # 对框进行进一步的处理
        new_boxes = np.array(self.merge_bboxes(box_datas, cutx, cuty))
        
        return mosaic_image, new_boxes
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        """合并Mosaic的边界框"""
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
    
    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        """MixUp数据增强"""
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
        # 使用动态步长
        grid_shapes = [input_shape // self.strides[l] for l in range(num_layers)]
        # 注意：y_true的shape应该是 [num_anchors, grid_h, grid_w, bbox_attrs]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / self.strides[l]
            
            # 计算真实框在特征图上的位置
            batch_target = np.zeros_like(targets)
            batch_target[:, [0,2]] = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[:, [1,3]] * in_h
            batch_target[:, 4] = targets[:, 4]
            
            # 过滤掉宽度或高度为0的边界框
            valid_mask = (batch_target[:, 2] > 0) & (batch_target[:, 3] > 0)
            if not np.any(valid_mask):
                continue
            
            valid_batch_target = batch_target[valid_mask]
            
            # 计算真实框和先验框的重合程度
            # 添加一个小的epsilon避免除零
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
                    
                    # 获得真实框属于哪个网格点
                    # 使用valid_batch_target而不是batch_target
                    i = int(np.floor(valid_batch_target[t, 0]))
                    j = int(np.floor(valid_batch_target[t, 1]))
                    
                    offsets = self.get_near_points(valid_batch_target[t, 0], valid_batch_target[t, 1], i, j)
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
                        
                        # 取出真实框的种类
                        c = int(valid_batch_target[t, 4])
                        
                        # tx、ty代表中心调整参数的真实值
                        y_true[l][k, local_j, local_i, 0] = valid_batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = valid_batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = valid_batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = valid_batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        
                        # 获得当前先验框最好的比例
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
        
        return y_true
    
    def get_near_points(self, x, y, i, j):
        """获取附近的点"""
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


def coco_dataset_collate(batch):
    """COCO数据集的collate函数"""
    images = []
    bboxes = []
    y_trues = [[] for _ in range(len(batch[0][2]))]
    
    for img, box, y_true in batch:
        images.append(img)
        bboxes.append(box)
        for i, y in enumerate(y_true):
            y_trues[i].append(y)
    
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann)).type(torch.FloatTensor) for ann in y_trues]
    
    return images, bboxes, y_trues
