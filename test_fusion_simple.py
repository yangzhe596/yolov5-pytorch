#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Fusion 转换器的完整流程（简化版，只处理一个序列）
"""

import sys
sys.path.insert(0, '/mnt/data/code/yolov5-pytorch')

import argparse

def test_fusion_conversion():
    """测试 Fusion 转换"""
    # 只测试序列 0
    args = argparse.Namespace(
        fred_root='/mnt/data/datasets/fred',
        output_root='test_output',
        split_mode='frame',
        threshold=0.033,
        train_ratio=1.0,  # 只用一个序列，全部用于测试
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
        simple=True,  # 简化模式
        simple_ratio=0.1,
        use_symlinks=False,
        validate_only=False
    )
    
    from convert_fred_to_fusion import FREDFusionConverter
    
    converter = FREDFusionConverter(
        fred_root=args.fred_root,
        output_root=args.output_root,
        split_mode=args.split_mode,
        threshold=args.threshold,
        simple=args.simple,
        simple_ratio=args.simple_ratio,
        use_symlinks=args.use_symlinks
    )
    
    print("="*70)
    print("测试 Fusion 转换 - 单序列")
    print("="*70)
    
    # 只处理序列 0
    sequences = [0]
    
    # 强制设置序列级别划分，只处理一个序列
    original_split_mode = converter.split_mode
    converter.split_mode = 'sequence'
    
    try:
        # 生成 train 划分
        coco_dict, stats = converter.generate_fusion_split(sequences, 'train')
        
        print(f"\n✅ 转换完成！")
        print(f"图像数量: {len(coco_dict['images'])}")
        print(f"标注数量: {len(coco_dict['annotations'])}")
        
        if coco_dict['images']:
            # 检查第一个图像
            first_img = coco_dict['images'][0]
            print(f"\n第一个图像信息:")
            print(f"  ID: {first_img['id']}")
            print(f"  file_name: {first_img.get('file_name', 'N/A')}")
            print(f"  rgb_file_name: {first_img.get('rgb_file_name', 'N/A')}")
            print(f"  event_file_name: {first_img.get('event_file_name', 'N/A')}")
            print(f"  width: {first_img.get('width', 'N/A')}")
            print(f"  height: {first_img.get('height', 'N/A')}")
            print(f"  modality: {first_img.get('modality', 'N/A')}")
        
        # 检查第一个标注
        if coco_dict['annotations']:
            first_ann = coco_dict['annotations'][0]
            print(f"\n第一个标注信息:")
            print(f"  ID: {first_ann['id']}")
            print(f"  image_id: {first_ann['image_id']}")
            print(f"  category_id: {first_ann['category_id']}")
            print(f"  bbox: {first_ann['bbox']}")
            print(f"  area: {first_ann['area']}")
            print(f"  drone_id: {first_ann.get('drone_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        converter.split_mode = original_split_mode


if __name__ == '__main__':
    try:
        success = test_fusion_conversion()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)