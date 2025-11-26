#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试帧级别划分验证
"""

from convert_fred_to_fusion_v2 import FrameSplitValidator

def debug_validation():
    """调试验证方法"""
    print("=" * 70)
    print("调试帧级别划分验证")
    print("=" * 70)
    
    validator = FrameSplitValidator(seed=42)
    
    # 使用已有的序列数据
    sequences = [1]  # 只测试序列 1
    
    print(f"\n测试序列: {sequences}")
    print(f"验证器 seed: {validator.seed}")
    
    # 创建模拟的帧数据
    simulated_data = {
        1: [
            {
                'rgb_timestamp': 100.0,
                'event_timestamp': 100.01,
                'rgb_path': 'rgb.jpg',
                'event_path': 'event.png',
                'status': 'dual'
            },
            {
                'rgb_timestamp': 101.0,
                'event_timestamp': 101.01,
                'rgb_path': 'rgb2.jpg',
                'event_path': 'event2.png',
                'status': 'dual'
            },
            {
                'rgb_timestamp': 102.0,
                'event_timestamp': 102.01,
                'rgb_path': 'rgb3.jpg',
                'event_path': 'event3.png',
                'status': 'dual'
            }
        ]
    }
    
    # 测试 get_frame_split
    print(f"\n测试 get_frame_split...")
    frame_info = {
        'rgb_timestamp': 100.0,
        'event_timestamp': 100.01
    }
    
    splits = []
    rand_vals = []
    for i in range(5):
        split, rand_val = validator.get_frame_split(1, frame_info)
        splits.append(split)
        rand_vals.append(rand_val)
        print(f"  第 {i+1} 次: split={split}, rand_val={rand_val:.6f}")
    
    if len(set(splits)) == 1 and len(set(rand_vals)) == 1:
        print("✅ 一致性检查通过（相同帧得到相同划分）")
    else:
        print("❌ 一致性检查失败！")
    
    # 测试不同帧
    print(f"\n测试不同帧...")
    different_splits = []
    for i, frame in enumerate(simulated_data[1]):
        split, rand_val = validator.get_frame_split(1, frame)
        different_splits.append(split)
        print(f"  帧 {i+1}: split={split}, rand_val={rand_val:.6f}")
    
    if len(set(different_splits)) > 1:
        print("✅ 不同帧得到不同划分（正常）")
    else:
        print("⚠️  所有帧都被分配到同一个划分（可能不正常，取决于 seed）")
    
    # 运行完整验证
    print(f"\n运行完整验证...")
    result = validator.validate_frame_split(sequences, num_checks=3)
    print(f"验证结果: {result}")

if __name__ == '__main__':
    debug_validation()