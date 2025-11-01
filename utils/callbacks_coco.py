"""
COCO格式数据集的评估回调
"""
import datetime
import os
import json
import shutil

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox


class CocoEvalCallback:
    """
    COCO格式数据集的评估回调
    支持从COCO JSON文件加载验证集
    """
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, 
                 coco_json_path, image_dir, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, 
                 nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1, max_eval_samples=None):
        """
        Args:
            net: 模型
            input_shape: 输入尺寸
            anchors: 先验框
            anchors_mask: 先验框mask
            class_names: 类别名称列表
            num_classes: 类别数量
            coco_json_path: COCO标注JSON文件路径
            image_dir: 图片目录
            log_dir: 日志目录
            cuda: 是否使用CUDA
            map_out_path: mAP计算临时目录
            max_boxes: 最大检测框数量
            confidence: 置信度阈值
            nms_iou: NMS的IOU阈值
            letterbox_image: 是否使用letterbox
            MINOVERLAP: mAP计算的IOU阈值
            eval_flag: 是否进行评估
            period: 评估周期（每多少个epoch评估一次）
            max_eval_samples: 最大评估样本数（用于快速验证）
        """
        super(CocoEvalCallback, self).__init__()
        
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
        
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, 
                                   (self.input_shape[0], self.input_shape[1]), 
                                   self.anchors_mask)
        
        self.maps = [0]
        self.epoches = [0]
        
        # 加载COCO标注
        self._load_coco_data()
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
    
    def _load_coco_data(self):
        """加载COCO格式的验证集数据"""
        print(f"加载验证集标注: {self.coco_json_path}")
        
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 创建category_id到索引的映射
        self.cat_id_to_idx = {}
        for cat in coco_data['categories']:
            self.cat_id_to_idx[cat['id']] = cat['id'] - 1
        
        # 创建image_id到annotations的映射
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # 保存图片信息
        self.images = coco_data['images']
        
        print(f"✓ 验证集加载完成: {len(self.images)} 张图片")
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """生成预测结果txt文件"""
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), 
                "w", encoding='utf-8')
        
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), 
                                 self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1), self.num_classes, self.input_shape,
                image_shape, self.letterbox_image, 
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )
            
            if results[0] is None:
                f.close()
                return
            
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        
        # 只保留top max_boxes个
        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], 
                str(int(left)), str(int(top)), 
                str(int(right)), str(int(bottom))
            ))
        
        f.close()
        return
    
    def on_epoch_end(self, epoch, model_eval):
        """Epoch结束时的回调"""
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            
            # 创建输出目录
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            
            print("Get map.")
            
            # 快速验证模式：限制评估样本数量
            eval_images = self.images
            if self.max_eval_samples is not None:
                eval_images = self.images[:self.max_eval_samples]
                print(f"⚡ 快速验证模式: 仅评估 {len(eval_images)} 个样本（共 {len(self.images)} 个）")
            
            # 遍历验证集
            for img_info in tqdm(eval_images, desc="Evaluating"):
                # 只使用文件名（不包含目录），避免路径问题
                file_name = img_info['file_name']
                image_id = os.path.splitext(os.path.basename(file_name))[0]
                img_path = os.path.join(self.image_dir, file_name)
                
                if not os.path.exists(img_path):
                    continue
                
                # 读取图像
                image = Image.open(img_path)
                
                # 生成预测结果
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                # 生成真实框txt
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), 
                         "w", encoding='utf-8') as new_f:
                    
                    # 从COCO标注中获取真实框
                    if img_info['id'] in self.img_to_anns:
                        for ann in self.img_to_anns[img_info['id']]:
                            bbox = ann['bbox']  # [x, y, width, height]
                            
                            # 转换为[left, top, right, bottom]
                            left = int(bbox[0])
                            top = int(bbox[1])
                            right = int(bbox[0] + bbox[2])
                            bottom = int(bbox[1] + bbox[3])
                            
                            # 获取类别
                            class_idx = self.cat_id_to_idx[ann['category_id']]
                            obj_name = self.class_names[class_idx]
                            
                            new_f.write("%s %s %s %s %s\n" % (
                                obj_name, left, top, right, bottom
                            ))
            
            print("Calculate Map.")
            
            try:
                # 尝试使用COCO API计算mAP
                from .utils_map import get_coco_map
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except Exception as e:
                # 如果COCO API不可用，使用VOC方式计算
                print(f"COCO mAP计算失败，使用VOC方式: {e}")
                from .utils_map import get_map
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            
            self.maps.append(temp_map)
            self.epoches.append(epoch)
            
            # 保存mAP记录
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            # 绘制mAP曲线
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')
            
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")
            
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")
            
            print(f"Epoch {epoch} mAP: {temp_map:.4f}")
            print("Get map done.")
            
            # 清理临时文件
            shutil.rmtree(self.map_out_path)


class SimplifiedEvalCallback:
    """
    简化版的评估回调
    只计算验证集loss，不计算mAP（节省时间）
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
        """Epoch结束时的回调"""
        if epoch % self.period == 0 and self.eval_flag:
            self.epoches.append(epoch)
            print(f"Epoch {epoch} 完成")
            # 这里可以添加其他评估逻辑
