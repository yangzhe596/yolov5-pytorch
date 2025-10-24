"""
COCO格式数据加载器
支持FRED数据集的COCO格式
"""
import json
import os
from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class CocoYoloDataset(Dataset):
    """COCO格式的YOLO数据集加载器"""
    
    def __init__(self, coco_json_path, image_dir, input_shape, num_classes, anchors, anchors_mask, 
                 epoch_length, mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
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
        
        self.epoch_now = -1
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4
        
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
                # 支持相对路径格式（如 "序列0/PADDED_RGB/xxx.jpg"）
                # image_dir 应该是 FRED 数据集的根目录
                img_path = os.path.join(self.image_dir, img['file_name'])
                
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
    
    def get_random_data(self, index, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """获取随机数据（单张图片）"""
        info = self.image_infos[index]
        
        # 读取图像
        image = Image.open(info['path'])
        image = cvtColor(image)
        
        # 获得图像的高宽与目标高宽
        iw, ih = image.size
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
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image = new_image
            
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
            
            return image, box
        
        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        
        # 翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data = np.array(image, np.uint8)
        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
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
        """Mosaic数据增强"""
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        
        image_datas = []
        box_datas = []
        index = 0
        
        for i in indices:
            info = self.image_infos[i]
            
            # 读取图像
            image = Image.open(info['path'])
            image = cvtColor(image)
            
            iw, ih = image.size
            box = info['boxes'].copy()
            
            # 翻转图像
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
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
            image = image.resize((nw, nh), Image.BICUBIC)
            
            # 将图像进行放置
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            elif index == 2:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            
            index = index + 1
            box_data = []
            
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
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
        
        # 将图像分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[2][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[3][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[1][:cuty, cutx:, :]
        
        new_image = np.array(new_image, np.uint8)
        
        # 对框进行进一步的处理
        new_boxes = np.array(self.merge_bboxes(box_datas, cutx, cuty))
        
        return new_image, new_boxes
    
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
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        # 注意：y_true的shape应该是 [num_anchors, grid_h, grid_w, bbox_attrs]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / {0:32, 1:16, 2:8}[l]
            
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
