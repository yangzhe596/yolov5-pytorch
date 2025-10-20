#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FRED数据集训练流程
快速验证所有组件是否正常工作
"""

import os
import sys

def test_imports():
    """测试所有必要的导入"""
    print("=" * 70)
    print("1. 测试导入")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        from utils.dataloader_coco import CocoYoloDataset, coco_dataset_collate
        print(f"✓ COCO数据加载器")
    except Exception as e:
        print(f"✗ COCO数据加载器导入失败: {e}")
        return False
    
    try:
        from nets.yolo import YoloBody
        print(f"✓ YOLO模型")
    except Exception as e:
        print(f"✗ YOLO模型导入失败: {e}")
        return False
    
    try:
        from utils.utils import get_anchors, get_classes
        print(f"✓ 工具函数")
    except Exception as e:
        print(f"✗ 工具函数导入失败: {e}")
        return False
    
    return True

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 70)
    print("2. 测试数据集加载")
    print("=" * 70)
    
    from utils.dataloader_coco import CocoYoloDataset
    from utils.utils import get_anchors
    
    # 检查数据集文件
    rgb_train = 'datasets/fred_coco/rgb/annotations/instances_train.json'
    event_train = 'datasets/fred_coco/event/annotations/instances_train.json'
    
    if not os.path.exists(rgb_train):
        print(f"✗ RGB训练集不存在: {rgb_train}")
        return False
    print(f"✓ RGB训练集存在")
    
    if not os.path.exists(event_train):
        print(f"✗ Event训练集不存在: {event_train}")
        return False
    print(f"✓ Event训练集存在")
    
    # 加载先验框
    try:
        anchors, _ = get_anchors('model_data/yolo_anchors.txt')
        print(f"✓ 先验框加载成功: {len(anchors)} 个")
    except Exception as e:
        print(f"✗ 先验框加载失败: {e}")
        return False
    
    # 测试RGB数据集
    try:
        rgb_dataset = CocoYoloDataset(
            rgb_train, 'datasets/fred_coco/rgb/train',
            [640, 640], 1, anchors, [[6,7,8],[3,4,5],[0,1,2]],
            100, False, False, 0, 0, False, 0
        )
        print(f"✓ RGB数据集加载成功: {len(rgb_dataset)} 张图片")
        
        # 测试加载一个样本
        image, box, y_true = rgb_dataset[0]
        print(f"  样本shape: {image.shape}, 边界框: {len(box)}")
    except Exception as e:
        print(f"✗ RGB数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试Event数据集
    try:
        event_dataset = CocoYoloDataset(
            event_train, 'datasets/fred_coco/event/train',
            [640, 640], 1, anchors, [[6,7,8],[3,4,5],[0,1,2]],
            100, False, False, 0, 0, False, 0
        )
        print(f"✓ Event数据集加载成功: {len(event_dataset)} 张图片")
        
        # 测试加载一个样本
        image, box, y_true = event_dataset[0]
        print(f"  样本shape: {image.shape}, 边界框: {len(box)}")
    except Exception as e:
        print(f"✗ Event数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 70)
    print("3. 测试模型创建")
    print("=" * 70)
    
    try:
        import torch
        from nets.yolo import YoloBody
        from utils.utils import get_anchors
        
        anchors, _ = get_anchors('model_data/yolo_anchors.txt')
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        
        model = YoloBody(
            anchors_mask=anchors_mask,
            num_classes=1,
            phi='s',
            backbone='cspdarknet',
            pretrained=False,
            input_shape=[640, 640]
        )
        
        print(f"✓ 模型创建成功")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ 前向传播成功")
        print(f"  输出层数: {len(outputs)}")
        print(f"  输出shapes: {[o.shape for o in outputs]}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """测试DataLoader"""
    print("\n" + "=" * 70)
    print("4. 测试DataLoader")
    print("=" * 70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from utils.dataloader_coco import CocoYoloDataset, coco_dataset_collate
        from utils.utils import get_anchors
        
        anchors, _ = get_anchors('model_data/yolo_anchors.txt')
        
        dataset = CocoYoloDataset(
            'datasets/fred_coco/rgb/annotations/instances_train.json',
            'datasets/fred_coco/rgb/train',
            [640, 640], 1, anchors, [[6,7,8],[3,4,5],[0,1,2]],
            100, False, False, 0, 0, False, 0
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=coco_dataset_collate,
            drop_last=True
        )
        
        print(f"✓ DataLoader创建成功")
        
        # 测试加载一个batch
        for images, boxes, y_trues in dataloader:
            print(f"✓ Batch加载成功")
            print(f"  Images shape: {images.shape}")
            print(f"  Boxes数量: {len(boxes)}")
            print(f"  Y_true层数: {len(y_trues)}")
            break
        
        return True
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("FRED数据集训练流程测试")
    print("=" * 70)
    print()
    
    results = []
    
    # 测试1: 导入
    results.append(("导入测试", test_imports()))
    
    # 测试2: 数据集加载
    if results[-1][1]:
        results.append(("数据集加载", test_dataset_loading()))
    else:
        print("\n跳过后续测试（导入失败）")
        results.append(("数据集加载", False))
    
    # 测试3: 模型创建
    if results[-1][1]:
        results.append(("模型创建", test_model_creation()))
    else:
        print("\n跳过后续测试（数据集加载失败）")
        results.append(("模型创建", False))
    
    # 测试4: DataLoader
    if results[-1][1]:
        results.append(("DataLoader", test_dataloader()))
    else:
        print("\n跳过后续测试（模型创建失败）")
        results.append(("DataLoader", False))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！可以开始训练。")
        print("=" * 70)
        print("\n下一步:")
        print("  /home/yz/miniforge3/envs/torch/bin/python3 train_fred.py --modality rgb")
    else:
        print("\n" + "=" * 70)
        print("❌ 部分测试失败，请检查错误信息。")
        print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
