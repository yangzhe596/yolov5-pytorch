#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试 simple 模式
"""

import random
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from convert_fred_to_fusion_v2 import FREDFusionConverterV2

def debug_simple_mode():
    """调试 simple 模式"""
    print("=" * 70)
    print("调试 simple 模式")
    print("=" * 70)
    
    # 创建转换器（启用 simple 模式）
    converter = FREDFusionConverterV2(
        fred_root='/mnt/data/datasets/fred',
        output_root='debug_output',
        split_mode='frame',
        simple=True,
        simple_ratio=0.1
    )
    
    print(f"\n转换器配置:")
    print(f"  simple: {converter.simple}")
    print(f"  simple_ratio: {converter.simple_ratio}")
    print(f"  seed: {converter.seed}")
    
    # 获取所有序列
    sequences = converter.get_all_sequences()
    print(f"\n所有序列 ({len(sequences)}): {sequences[:10]}...{sequences[-5:]}")
    
    # 测试采样
    print(f"\n测试采样...")
    for i in range(3):
        sampled = converter._sample_sequences(sequences)
        print(f"第 {i+1} 次采样: {len(sampled)} 个序列 - {sampled}")
    
    # 测试大比例采样
    print(f"\n测试大比例采样 (100%):")
    converter.simple_ratio = 1.0
    sampled = converter._sample_sequences(sequences)
    print(f"采样结果: {len(sampled)} == {len(sequences)} ? {len(sampled) == len(sequences)}")
    
    # 测试小比例采样
    print(f"\n测试小比例采样 (2%):")
    converter.simple_ratio = 0.02
    sampled = converter._sample_sequences(sequences)
    print(f"采样结果: {len(sampled)} 序列 - {sampled}")
    
    print("\n✅ 调试完成")

if __name__ == '__main__':
    debug_simple_mode()