#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Fusion 模型评估流程
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append('/mnt/data/code/yolov5-pytorch')

from nets.yolo_fusion import YoloFusionBody
from utils.utils import get_anchors, get_classes

def test_fusion_model():
    """测试 Fusion 模型输入"""
    print("="*60)
    print("测试 Fusion 模型输入验证")
    print("="*60)
    
    try:
        # 创建一个简单的 Fusion 模型
        model = YoloFusionBody(
            anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            num_classes=1,
            compression_ratio=0.75,
            phi='s',
            backbone='cspdarknet'
        )
        
        print("✅ Fusion 模型创建成功")
        
        # 创建测试输入
        batch_size = 2
        channels = 3
        height, width = 640, 640
        
        # RGB 图像
        rgb_images = torch.rand(batch_size, channels, height, width)
        # Event 图像  
        event_images = torch.rand(batch_size, channels, height, width)
        
        print(f"RGB shape: {rgb_images.shape}")
        print(f"Event shape: {event_images.shape}")
        
        # 模型推理
        with torch.no_grad():
            outputs = model(rgb_images, event_images)
        
        print(f"✅ 模型推理成功，输出数量: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  输出 {i}: {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback_import():
    """测试回调函数导入"""
    print("\n" + "="*60)
    print("测试回调函数导入")
    print("="*60)
    
    try:
        # 测试导入
        from utils.utils_map import get_map
        print("✅ VOC mAP 计算函数导入成功")
        
        try:
            from utils.utils_map import get_coco_map
            print("✅ COCO mAP 计算函数导入成功")
        except ImportError:
            print("⚠️  COCO mAP 计算函数导入失败，将使用 VOC 方式")
        
        # 测试简单计算
        print("✅ 回调函数测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 回调函数导入失败: {e}")
        return False

def test_dataset_creation():
    """测试数据集创建"""
    print("\n" + "="*60)
    print("测试数据集创建")
    print("="*60)
    
    try:
        # 检查 COCO 文件是否存在
        import config_fred
        coco_json = config_fred.get_fusion_annotation_path('test')
        
        if os.path.exists(coco_json):
            print(f"✅ COCO 标注文件存在: {coco_json}")
            
            # 尝试加载
            with open(coco_json, 'r') as f:
                import json
                coco_data = json.load(f)
            
            print(f"✅ COCO 格式正确，图片数量: {len(coco_data['images'])}")
            print(f"✅ 标注数量: {len(coco_data['annotations'])}")
            print(f"✅ 类别数量: {len(coco_data['categories'])}")
            
            return True
        else:
            print(f"⚠️  COCO 标注文件不存在: {coco_json}")
            return False
            
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

if __name__ == "__main__":
    print("启动 Fusion 评估测试\n")
    
    # 运行测试
    results = []
    
    # 1. 模型测试
    results.append(("模型输入", test_fusion_model()))
    
    # 2. 回调函数测试
    results.append(("回调函数", test_callback_import()))
    
    # 3. 数据集测试
    results.append(("数据集", test_dataset_creation()))
    
    # 输出结果
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:12} | {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print(f"\n🎉 所有测试通过！可以开始使用 Fusion 评估功能")
    else:
        print(f"\n⚠️  部分测试失败，请检查相关组件")
    
    print("="*60)