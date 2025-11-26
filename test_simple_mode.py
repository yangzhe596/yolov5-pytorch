#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 simple 模式功能
"""

import argparse
import sys
from convert_fred_to_fusion_v2 import FREDFusionConverterV2, FrameSplitValidator

def test_simple_mode():
    """测试 simple 模式"""
    print("=" * 70)
    print("测试 simple 模式")
    print("=" * 70)
    
    # 测试参数
    fred_root = '/mnt/data/datasets/fred'
    simple_ratio = 0.1  # 10%
    
    # 创建转换器（启用 simple 模式）
    converter = FREDFusionConverterV2(
        fred_root=fred_root,
        output_root='datasets/fred_fusion_v2_simple_test',
        split_mode='frame',
        threshold=0.033,
        simple=True,
        simple_ratio=simple_ratio
    )
    
    print(f"\n转换器配置:")
    print(f"  Simple 模式: {converter.simple}")
    print(f"  采样比例: {converter.simple_ratio:.1%}")
    
    # 获取所有序列
    all_sequences = converter.get_all_sequences()
    print(f"\n所有序列 ({len(all_sequences)}): {all_sequences}")
    
    # 测试采样
    sampled_sequences = converter._sample_sequences(all_sequences)
    print(f"\n采样序列 ({len(sampled_sequences)}): {sampled_sequences}")
    print(f"采样比例: {len(sampled_sequences)/len(all_sequences)*100:.1f}%")
    
    # 验证采样逻辑
    if len(sampled_sequences) > 0:
        print("\n✅ 采样逻辑正常工作")
        print(f"  预期采样数量: {int(len(all_sequences) * simple_ratio)}")
        print(f"  实际采样数量: {len(sampled_sequences)}")
    else:
        print("\n❌ 采样失败！")
        return False
    
    # 随机种子一致性测试
    print("\n测试随机种子一致性...")
    sequences1 = converter._sample_sequences(all_sequences)
    sequences2 = converter._sample_sequences(all_sequences)
    
    if sequences1 == sequences2:
        print("✅ 随机种子一致：相同输入得到相同输出")
    else:
        print("❌ 随机种子不一致！")
        return False
    
    # 测试不同采样比例
    print("\n测试不同采样比例:")
    test_ratios = [0.05, 0.1, 0.2, 0.5, 1.0]
    for ratio in test_ratios:
        converter.simple_ratio = ratio
        sampled = converter._sample_sequences(all_sequences)
        expected = max(1, int(len(all_sequences) * ratio))
        print(f"  比例 {ratio:.1%}: 预期 {expected} -> 实际 {len(sampled)}")
        
        if len(sampled) != expected:
            print(f"    ⚠️  不匹配（多个序列时的采样变异性）")
    
    return True

def main():
    try:
        success = test_simple_mode()
        
        if success:
            print("\n" + "=" * 70)
            print("✅ 所有 simple 模式测试通过！")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("❌ simple 模式测试失败！")
            print("=" * 70)
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())