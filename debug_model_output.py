# -*- coding: utf-8 -*-
"""
调试 Fusion 模型输出维度
"""

import sys
sys.path.append('.')

import torch
from nets.yolo_fusion import YoloFusionBody
from nets.yolo import YoloBody
from utils.dataloader_fusion import FusionYoloDataset, fusion_dataset_collate
import numpy as np

print("="*80)
print("测试 Fusion 模型输出维度")
print("="*80)

# 创建模型
fusion_model = YoloFusionBody(
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    num_classes=1,
    compression_ratio=1.0,
    phi='s',
    backbone='cspdarknet'
)

# 创建模拟输入
batch_size = 2
rgb_image = torch.randn(batch_size, 3, 640, 640)
event_image = torch.randn(batch_size, 3, 640, 640)

print("输入形状:")
print(f"  RGB 图像: {rgb_image.shape}")
print(f"  Event 图像: {event_image.shape}")

# 前向传播
with torch.no_grad():
    outputs = fusion_model(rgb_image, event_image)

print("\nFusion 模型输出形状:")
for i, output in enumerate(outputs):
    print(f"  输出 {i}: {output.shape}")

# 创建单模态模型进行对比
single_model = YoloBody(
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    num_classes=1,
    backbone='cspdarknet',
    phi='s'
)

print("\n单模态模型输出形状:")
with torch.no_grad():
    single_outputs = single_model(rgb_image)
for i, output in enumerate(single_outputs):
    print(f"  输出 {i}: {output.shape}")

# 检查维度是否一致
print("\n维度对比:")
for i in range(len(outputs)):
    if outputs[i].shape == single_outputs[i].shape:
        print(f"  输出 {i}: ✅ 维度一致")
    else:
        print(f"  输出 {i}: ❌ 维度不一致")
        print(f"    Fusion: {outputs[i].shape}")
        print(f"    单模态: {single_outputs[i].shape}")

print("\n" + "="*80)
print("测试完成")
print("="*80)