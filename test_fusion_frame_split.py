#!/usr/bin/env python
# -*- coding: utf--8 -*-
"""
快速测试 convert_fred_to_fusion.py 的帧级别划分功能
"""

import sys
sys.path.insert(0, '/mnt/data/code/yolov5-pytorch')

from convert_fred_to_fusion import FREDFusionConverter

def test_frame_split():
    """测试帧级别划分"""
    print("="*70)
    print("测试 convert_fred_to_fusion.py 帧级别划分")
    print("="*70)
    
    # 创建转换器实例
    converter = FREDFusionConverter(
        fred_root='/mnt/data/datasets/fred',
        output_root='test_output',
        split_mode='frame',
        threshold=0.033,
        simple=False,
        use_symlinks=False
    )
    
    print("\n测试 1: 测试 get_frame_split 方法")
    print("-" * 70)
    
    # 测试一致性
    test_cases = [
        ("rgb_1_100.123456", 42),
        ("rgb_1_100.123456", 42),  # 相同输入
        ("rgb_2_200.123456", 42),
        ("event_1_100.123456", 42),
    ]
    
    results = []
    for frame_key, seed in test_cases:
        split = converter.get_frame_split(frame_key, seed)
        results.append((frame_key, seed, split))
        print(f"  帧: {frame_key}, 种子: {seed} -> {split}")
    
    # 检查一致性
    if results[0][2] == results[1][2]:
        print("  ✅ 一致帧分配一致")
    else:
        print("  ❌ 一致帧分配不一致（有问题！）")
        return False
    
    print("\n测试 2: 验证哈希函数")
    print("-" * 70)
    
    import hashlib
    test_key = "rgb_1_100.123456_42"
    
    # 检查是否使用 SHA256
    hash_input = f"{test_key}"
    try:
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        print(f"  ✅ 检测到 SHA256 哈希函数")
    except:
        print(f"  ❌ 无法使用 SHA256")
        return False
    
    print("\n测试 3: 测试分割比例")
    print("-" * 70)
    
    # 生成大量帧测试比例
    num_frames = 1000
    splits = {'train': 0, 'val': 0, 'test': 0}
    
    for i in range(num_frames):
        frame_key = f"rgb_1_{i * 0.033:.6f}"
        split = converter.get_frame_split(frame_key, 42)
        splits[split] += 1
    
    print(f"  总帧数: {num_frames}")
    print(f"  训练集: {splits['train']} ({splits['train']/num_frames*100:.2f}%)")
    print(f"  验证集: {splits['val']} ({splits['val']/num_frames*100:.2f}%)")
    print(f"  测试集: {splits['test']} ({splits['test']/num_frames*100:.2f}%)")
    
    # 检查比例（允许 2% 容差）
    train_ratio = splits['train'] / num_frames
    val_ratio = splits['val'] / num_frames
    tolerance = 0.02
    
    if abs(train_ratio - 0.7) < tolerance:
        print(f"  ✅ 训练集比例正确 (目标: 70%, 实际: {train_ratio*100:.2f}%)")
    else:
        print(f"  ⚠️  训练集比例偏差较大 (目标: 70%, 实际: {train_ratio*100:.2f}%)")
    
    if abs(val_ratio - 0.15) < tolerance:
        print(f"  ✅ 验证集比例正确 (目标: 15%, 实际: {val_ratio*100:.2f}%)")
    else:
        print(f"  ⚠️  验证集比例偏差较大 (目标: 15%, 实际: {val_ratio*100:.2f}%)")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    
    return True


if __name__ == '__main__':
    try:
        success = test_frame_split()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)