#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已训练的FRED模型进行mAP评估
"""
import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image

from nets.yolo import YoloBody
from utils.utils import get_anchors, cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox
from utils.utils_map import get_coco_map, get_map


def evaluate_model(model_path, modality='rgb', coco_root='datasets/fred_coco', 
                   confidence=0.05, nms_iou=0.5, input_shape=(640, 640),
                   map_out_path='map_out', MINOVERLAP=0.5):
    """
    评估已训练的模型
    
    Args:
        model_path: 模型权重路径
        modality: 模态 (rgb/event)
        coco_root: COCO数据集根目录
        confidence: 置信度阈值
        nms_iou: NMS IOU阈值
        input_shape: 输入尺寸
        map_out_path: mAP计算输出目录
        MINOVERLAP: mAP计算的IOU阈值
    """
    
    # 设备配置
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    # 数据集配置
    coco_path = os.path.join(coco_root, modality)
    test_json = os.path.join(coco_path, 'annotations', 'instances_test.json')
    # 使用FRED数据集根目录（file_name包含相对路径）
    fred_root = '/home/yz/datasets/fred'
    test_img_dir = fred_root
    
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"测试集标注文件不存在: {test_json}")
    
    # 模型配置
    num_classes = 1
    class_names = ['object']
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    backbone = 'cspdarknet'
    phi = 's'
    
    print("=" * 70)
    print(f"FRED {modality.upper()} 模型评估")
    print("=" * 70)
    print(f"模型权重: {model_path}")
    print(f"测试集: {test_json}")
    print(f"输入尺寸: {input_shape}")
    print(f"置信度阈值: {confidence}")
    print(f"NMS IOU阈值: {nms_iou}")
    print(f"mAP IOU阈值: {MINOVERLAP}")
    print("=" * 70)
    
    # 加载先验框
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 创建模型
    print("\n加载模型...")
    model = YoloBody(anchors_mask, num_classes, phi, backbone, 
                     pretrained=False, input_shape=input_shape)
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
    
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    
    # 处理可能的DataParallel包装
    if list(pretrained_dict.keys())[0].startswith('module.'):
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    
    print(f"✓ 成功加载 {len(load_key)} 个参数")
    if len(no_load_key) > 0:
        print(f"⚠ 未加载 {len(no_load_key)} 个参数")
    
    model.eval()
    if cuda:
        model = model.cuda()
    
    # 创建解码器
    bbox_util = DecodeBox(anchors, num_classes, 
                         (input_shape[0], input_shape[1]), anchors_mask)
    
    # 加载COCO测试集
    print("\n加载测试集...")
    with open(test_json, 'r') as f:
        coco_data = json.load(f)
    
    # 创建category_id到索引的映射
    cat_id_to_idx = {}
    for cat in coco_data['categories']:
        cat_id_to_idx[cat['id']] = cat['id'] - 1
    
    # 创建image_id到annotations的映射
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    images = coco_data['images']
    print(f"✓ 测试集: {len(images)} 张图片")
    
    # 创建输出目录
    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, "ground-truth"), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, "detection-results"), exist_ok=True)
    
    print("\n开始评估...")
    
    # 遍历测试集
    for img_info in tqdm(images, desc="处理图片"):
        # 只使用文件名（不包含目录），避免路径问题
        file_name = img_info['file_name']
        image_id = os.path.splitext(os.path.basename(file_name))[0]
        img_path = os.path.join(test_img_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"⚠ 图片不存在: {img_path}")
            continue
        
        # 读取图像
        image = Image.open(img_path)
        image_shape = np.array(np.shape(image)[0:2])
        
        # 预处理
        image_rgb = cvtColor(image)
        image_data = resize_image(image_rgb, (input_shape[1], input_shape[0]), True)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0
        )
        
        # 预测
        with torch.no_grad():
            images_tensor = torch.from_numpy(image_data)
            if cuda:
                images_tensor = images_tensor.cuda()
            
            outputs = model(images_tensor)
            outputs = bbox_util.decode_box(outputs)
            results = bbox_util.non_max_suppression(
                torch.cat(outputs, 1), num_classes, input_shape,
                image_shape, True, conf_thres=confidence, nms_thres=nms_iou
            )
        
        # 保存预测结果
        det_file = os.path.join(map_out_path, "detection-results", f"{image_id}.txt")
        with open(det_file, "w", encoding='utf-8') as f:
            if results[0] is not None:
                top_label = np.array(results[0][:, 6], dtype='int32')
                top_conf = results[0][:, 4] * results[0][:, 5]
                top_boxes = results[0][:, :4]
                
                for i, c in enumerate(top_label):
                    predicted_class = class_names[int(c)]
                    box = top_boxes[i]
                    score = top_conf[i]
                    
                    top, left, bottom, right = box
                    f.write(f"{predicted_class} {score:.6f} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")
        
        # 保存真实框
        gt_file = os.path.join(map_out_path, "ground-truth", f"{image_id}.txt")
        with open(gt_file, "w", encoding='utf-8') as f:
            if img_info['id'] in img_to_anns:
                for ann in img_to_anns[img_info['id']]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # 转换为[left, top, right, bottom]
                    left = int(bbox[0])
                    top = int(bbox[1])
                    right = int(bbox[0] + bbox[2])
                    bottom = int(bbox[1] + bbox[3])
                    
                    # 获取类别
                    class_idx = cat_id_to_idx[ann['category_id']]
                    obj_name = class_names[class_idx]
                    
                    f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
    
    # 计算mAP
    print("\n计算mAP...")
    try:
        # 尝试使用COCO API
        from pycocotools.coco import COCO
        temp_map = get_coco_map(class_names=class_names, path=map_out_path)[1]
        print(f"\n{'='*70}")
        print(f"mAP@{MINOVERLAP} (COCO): {temp_map:.4f}")
        print(f"{'='*70}")
    except Exception as e:
        # 使用VOC方式计算
        print(f"使用VOC方式计算mAP (COCO API不可用)")
        temp_map = get_map(MINOVERLAP, False, path=map_out_path)
        print(f"\n{'='*70}")
        print(f"mAP@{MINOVERLAP} (VOC): {temp_map:.4f}")
        print(f"{'='*70}")
    
    # 保存结果
    result_file = os.path.join(os.path.dirname(model_path), f"eval_result_{modality}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Set: {test_json}\n")
        f.write(f"mAP@{MINOVERLAP}: {temp_map:.4f}\n")
    
    print(f"\n✓ 评估结果已保存到: {result_file}")
    
    return temp_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估FRED训练的YOLOv5模型')
    parser.add_argument('--model_path', type=str, default='logs/fred_rgb/best_epoch_weights.pth',
                        help='模型权重路径')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='模态: rgb 或 event')
    parser.add_argument('--coco_root', type=str, default='datasets/fred_coco',
                        help='COCO数据集根目录')
    parser.add_argument('--confidence', type=float, default=0.05,
                        help='置信度阈值')
    parser.add_argument('--nms_iou', type=float, default=0.5,
                        help='NMS IOU阈值')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[640, 640],
                        help='输入尺寸 (height width)')
    parser.add_argument('--map_out_path', type=str, default='map_out',
                        help='mAP计算输出目录')
    parser.add_argument('--minoverlap', type=float, default=0.5,
                        help='mAP计算的IOU阈值')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        modality=args.modality,
        coco_root=args.coco_root,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        input_shape=tuple(args.input_shape),
        map_out_path=args.map_out_path,
        MINOVERLAP=args.minoverlap
    )
