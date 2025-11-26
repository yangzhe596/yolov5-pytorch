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

# 导入FRED配置
import config_fred as cfg


def evaluate_model(model_path, modality='rgb', 
                   confidence=None, nms_iou=None, input_shape=None,
                   map_out_path='map_out', MINOVERLAP=None,
                   high_res=False, four_features=False):
    """
    评估已训练的模型
    
    Args:
        model_path: 模型权重路径
        modality: 模态 (rgb/event)
        confidence: 置信度阈值（None则使用配置文件）
        nms_iou: NMS IOU阈值（None则使用配置文件）
        input_shape: 输入尺寸（None则使用配置文件）
        map_out_path: mAP计算输出目录
        MINOVERLAP: mAP计算的IOU阈值（None则使用配置文件）
        high_res: 是否使用高分辨率模式
        four_features: 是否使用四特征层模式（需要high_res=True）
    """
    
    # 从配置文件加载参数（如果未指定）
    if confidence is None:
        confidence = cfg.CONFIDENCE
    if nms_iou is None:
        nms_iou = cfg.NMS_IOU
    if input_shape is None:
        input_shape = tuple(cfg.INPUT_SHAPE)
    if MINOVERLAP is None:
        MINOVERLAP = cfg.MINOVERLAP
    
    # 设备配置
    cuda = cfg.CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    # 数据集配置
    test_json = cfg.get_annotation_path(modality, 'test')
    test_img_dir = cfg.get_image_dir(modality)
    
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"测试集标注文件不存在: {test_json}")
    
    # 模型配置
    num_classes = cfg.NUM_CLASSES
    class_names = cfg.CLASS_NAMES
    anchors_path = cfg.ANCHORS_PATH
    anchors_mask = cfg.ANCHORS_MASK
    backbone = cfg.BACKBONE
    phi = cfg.PHI
    
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
                     pretrained=False, input_shape=input_shape,
                     high_res=high_res, four_features=four_features)
    
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
                         (input_shape[0], input_shape[1]), anchors_mask, 
                         high_res=high_res, four_features=four_features)
    
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
    
    # 统计缺失图片
    missing_images = 0
    
    # 遍历测试集
    for img_info in tqdm(images, desc="处理图片"):
        # 只使用文件名（不包含目录），避免路径问题
        file_name = img_info['file_name']
        image_id = os.path.splitext(os.path.basename(file_name))[0]
        img_path = os.path.join(test_img_dir, file_name)
        
        if not os.path.exists(img_path):
            missing_images += 1
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
    
    # 报告缺失图片
    if missing_images > 0:
        print(f"\n⚠ 警告: {missing_images} 张图片未找到")
    
    # 计算mAP
    print("\n计算mAP...")
    # 首先检查pycocotools是否可用
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        # 如果导入成功，尝试使用COCO API
        temp_map = get_coco_map(class_names=class_names, path=map_out_path)[1]
        print(f"\n{'='*70}")
        print(f"mAP@{MINOVERLAP} (COCO): {temp_map:.4f}")
        print(f"{'='*70}")
    except (ImportError, NameError, Exception) as e:
        # 使用VOC方式计算
        print(f"使用VOC方式计算mAP (COCO API不可用: {str(e)})")
        temp_map = get_map(MINOVERLAP, False, path=map_out_path)
        print(f"\n{'='*70}")
        print(f"mAP@{MINOVERLAP} (VOC): {temp_map:.4f}")
        print(f"{'='*70}")
    
    # 保存结果
    save_dir = cfg.get_save_dir(modality)
    result_file = os.path.join(save_dir, f"eval_result_{modality}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Set: {test_json}\n")
        f.write(f"mAP@{MINOVERLAP}: {temp_map:.4f}\n")
    
    print(f"\n✓ 评估结果已保存到: {result_file}")
    
    return temp_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估FRED训练的YOLOv5模型')
    parser.add_argument('--model_path', type=str, default='',
                        help='模型权重路径（默认使用配置文件中的最佳权重）')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'],
                        help='模态: rgb 或 event')
    parser.add_argument('--confidence', type=float, default=None,
                        help=f'置信度阈值（默认: {cfg.CONFIDENCE}）')
    parser.add_argument('--nms_iou', type=float, default=None,
                        help=f'NMS IOU阈值（默认: {cfg.NMS_IOU}）')
    parser.add_argument('--input_shape', type=int, nargs=2, default=None,
                        help=f'输入尺寸 (height width)（默认: {cfg.INPUT_SHAPE}）')
    parser.add_argument('--map_out_path', type=str, default='map_out',
                        help='mAP计算输出目录')
    parser.add_argument('--minoverlap', type=float, default=None,
                        help=f'mAP计算的IOU阈值（默认: {cfg.MINOVERLAP}）')
    parser.add_argument('--high_res', action='store_true',
                        help='使用高分辨率模式模型')
    parser.add_argument('--four_features', action='store_true',
                        help='使用四特征层模式模型（需要同时指定--high_res）')
    
    args = parser.parse_args()
    
    # 配置高分辨率模式
    if args.high_res:
        cfg.configure_high_res_mode(True, args.four_features)
        print(f"\n{'='*70}")
        if args.four_features:
            print("🔍 四特征层高分辨率模式已启用")
            print("  - 特征层: 160x160, 80x80, 40x40, 20x20")
        else:
            print("🔍 高分辨率模式已启用")
            print("  - 特征层: 160x160, 80x80, 40x40")
        print(f"{'='*70}\n")
    
    # 如果未指定模型路径，使用配置文件中的最佳权重
    if not args.model_path:
        args.model_path = cfg.get_model_path(args.modality, best=True)
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        print(f"\n请先训练模型:")
        if args.four_features:
            print(f"  python train_fred.py --modality {args.modality} --high_res --four_features")
        elif args.high_res:
            print(f"  python train_fred.py --modality {args.modality} --high_res")
        else:
            print(f"  python train_fred.py --modality {args.modality}")
        exit(1)
    
    # 转换input_shape为tuple
    input_shape = tuple(args.input_shape) if args.input_shape else None
    
    evaluate_model(
        model_path=args.model_path,
        modality=args.modality,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        input_shape=input_shape,
        map_out_path=args.map_out_path,
        MINOVERLAP=args.minoverlap,
        high_res=args.high_res,
        four_features=args.four_features
    )
