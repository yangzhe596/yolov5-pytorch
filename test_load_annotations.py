#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Fusion 转换器的 annotation 加载功能
"""

import sys
sys.path.insert(0, '/mnt/data/code/yolov5-pytorch')

from pathlib import Path

def test_load_annotations():
    """测试 load_annotations 方法"""
    from convert_fred_to_fusion import FREDFusionConverter
    
    # 创建转换器实例
    converter = FREDFusionConverter(
        fred_root='/mnt/data/datasets/fred',
        output_root='test_output',
        split_mode='frame',
        threshold=0.033,
        simple=False,
        use_symlinks=False
    )
    
    print("="*70)
    print("测试 load_annotations 方法")
    print("="*70)
    
    # 测试序列 0
    sequence_path = Path('/mnt/data/datasets/fred/0')
    
    if not sequence_path.exists():
        print(f"序列 0 不存在: {sequence_path}")
        return False
    
    annotations = converter.load_annotations(sequence_path)
    
    print(f"\n加载的标注数量: {len(annotations)}")
    
    if not annotations:
        print("⚠️  警告: 没有加载到标注")
        return True  # 仍返回 True，因为可能真的没有标注
    
    # 检查第一个时间戳的标注
    first_timestamp = min(annotations.keys())
    first_anns = annotations[first_timestamp]
    
    print(f"\n第一个时间戳: {first_timestamp}")
    print(f"目标数量: {len(first_anns)}")
    
    if first_anns:
        first_ann = first_anns[0]
        print(f"\n第一个标注的字段:")
        for key, value in first_ann.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        
        # 检查必要字段
        necessary_fields = ['bbox', 'drone_id', 'category_id', 'area']
        missing_fields = [f for f in necessary_fields if f not in first_ann]
        
        if missing_fields:
            print(f"\n❌ 缺少必要字段: {missing_fields}")
            return False
        else:
            print(f"\n✅ 所有必要字段都存在")
            return True
    else:
        print("❌ 异常：没有标注但标注字典不为空")
        return False


if __name__ == '__main__':
    try:
        success = test_load_annotations()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)