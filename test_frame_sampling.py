#!/usr/bin/env python3
"""
测试帧级别简化模式
验证 --simple 参数现在是在所有帧中随机挑选固定比例的帧，而不是随机挑选视频序列
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convert_fred_to_fusion import FREDFusionConverterV2
from collections import defaultdict

def test_frame_level_sampling():
    """测试帧级别采样逻辑"""
    print("=" * 70)
    print("测试帧级别简化模式")
    print("=" * 70)
    
    # 创建转换器（使用实际数据集路径）
    converter = FREDFusionConverterV2(
        fred_root="/mnt/data/datasets/fred",
        output_root="/mnt/data/code/yolov5-pytorch/datasets/fred_fusion_test",
        split_mode="frame",
        simple=True,
        simple_ratio=0.1  # 10% 的帧
    )
    
    # 获取所有序列
    sequences = converter.get_all_sequences()
    print(f"找到 {len(sequences)} 个序列")
    
    # 模拟帧数据
    all_sequences_data = {}
    total_frames = 0
    
    for seq_id in sequences[:5]:  # 只用前5个序列测试
        frames = []
        # 每个序列模拟20帧
        for i in range(20):
            frame_info = {
                'rgb_timestamp': 1000.0 + i * 33.0,  # 33ms 间隔
                'rgb_path': f"/fake/rgb/{seq_id}/frame_{i:04d}.jpg",
                'event_timestamp': 1000.0 + i * 33.0 + 0.5,  # 稍微偏移
                'event_path': f"/fake/event/{seq_id}/frame_{i:04d}.png",
                'time_diff': 0.5,
                'status': 'dual'
            }
            frames.append(frame_info)
        
        all_sequences_data[seq_id] = frames
        total_frames += len(frames)
    
    print(f"模拟数据: {len(all_sequences_data)} 个序列, {total_frames} 帧")
    
    # 测试帧级别采样
    sampled_data = converter._sample_frames(all_sequences_data)
    
    # 统计采样结果
    sampled_frames = sum(len(frames) for frames in sampled_data.values())
    expected_frames = max(1, int(total_frames * converter.simple_ratio))
    
    print(f"\n采样结果:")
    print(f"  原始帧数: {total_frames}")
    print(f"  采样帧数: {sampled_frames}")
    print(f"  预期帧数: {expected_frames}")
    print(f"  实际比例: {sampled_frames/total_frames:.3f}")
    print(f"  目标比例: {converter.simple_ratio:.3f}")
    
    # 验证每个序列的采样情况
    print(f"\n各序列采样情况:")
    for seq_id in all_sequences_data:
        original_count = len(all_sequences_data[seq_id])
        sampled_count = len(sampled_data.get(seq_id, []))
        print(f"  {seq_id}: {original_count} -> {sampled_count} 帧")
    
    # 测试一致性：多次采样应得到相同结果
    print(f"\n测试采样一致性...")
    sampled_data_2 = converter._sample_frames(all_sequences_data)
    
    consistent = True
    for seq_id in sampled_data:
        if seq_id not in sampled_data_2:
            consistent = False
            break
        
        frames1 = sampled_data[seq_id]
        frames2 = sampled_data_2[seq_id]
        
        if len(frames1) != len(frames2):
            consistent = False
            break
        
        # 检查时间戳是否相同
        timestamps1 = sorted([f['rgb_timestamp'] for f in frames1])
        timestamps2 = sorted([f['rgb_timestamp'] for f in frames2])
        
        if timestamps1 != timestamps2:
            consistent = False
            break
    
    if consistent:
        print(f"✅ 采样一致性测试通过")
    else:
        print(f"❌ 采样一致性测试失败")
    
    return sampled_frames >= expected_frames * 0.9 and sampled_frames <= expected_frames * 1.1 and consistent

def test_vs_sequence_sampling():
    """对比帧级别采样和序列级别采样的差异"""
    print("\n" + "=" * 70)
    print("对比帧级别采样 vs 序列级别采样")
    print("=" * 70)
    
    # 创建模拟数据
    sequences = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5']
    all_sequences_data = {}
    
    # 创建不同长度的序列
    sequence_lengths = [10, 20, 30, 40, 50]  # 不同序列有不同数量的帧
    
    for seq_id, length in zip(sequences, sequence_lengths):
        frames = []
        for i in range(length):
            frame_info = {
                'rgb_timestamp': 1000.0 + i * 33.0,
                'rgb_path': f"/fake/rgb/{seq_id}/frame_{i:04d}.jpg",
                'event_timestamp': 1000.0 + i * 33.0 + 0.5,
                'event_path': f"/fake/event/{seq_id}/frame_{i:04d}.png",
                'time_diff': 0.5,
                'status': 'dual'
            }
            frames.append(frame_info)
        all_sequences_data[seq_id] = frames
    
    total_frames = sum(sequence_lengths)
    print(f"模拟数据: {len(sequences)} 个序列, {total_frames} 帧")
    print(f"序列长度分布: {sequence_lengths}")
    
    # 帧级别采样
    converter_frame = FREDFusionConverterV2(
        fred_root="/mnt/data/datasets/fred",
        output_root="/mnt/data/code/yolov5-pytorch/datasets/fred_fusion_test",
        split_mode="frame",
        simple=True,
        simple_ratio=0.2  # 20%
    )
    
    sampled_frame_data = converter_frame._sample_frames(all_sequences_data)
    sampled_frame_count = sum(len(frames) for frames in sampled_frame_data.values())
    
    # 模拟序列级别采样（旧逻辑）
    def sample_sequences_old(sequences, ratio=0.2):
        import random
        random.seed(42)
        n_sample = max(1, int(len(sequences) * ratio))
        sampled = random.sample(sequences, n_sample)
        return sorted(sampled)
    
    sampled_sequences = sample_sequences_old(sequences, 0.2)
    sampled_seq_count = sum(sequence_lengths[sequences.index(seq)] for seq in sampled_sequences)
    
    print(f"\n采样结果对比:")
    print(f"  帧级别采样: {sampled_frame_count} 帧 ({sampled_frame_count/total_frames:.1%})")
    print(f"  序列级别采样: {sampled_seq_count} 帧 ({sampled_seq_count/total_frames:.1%})")
    print(f"  采样序列: {sampled_sequences}")
    
    # 分析帧级别采样的分布
    print(f"\n帧级别采样分布:")
    for seq_id in all_sequences_data:
        original = len(all_sequences_data[seq_id])
        sampled = len(sampled_frame_data.get(seq_id, []))
        ratio = sampled / original if original > 0 else 0
        print(f"  {seq_id}: {sampled}/{original} 帧 ({ratio:.1%})")

if __name__ == "__main__":
    print("测试修改后的简化模式逻辑")
    print("现在 --simple 应该是在所有帧中随机挑选固定比例的帧")
    print()
    
    # 运行测试
    success = test_frame_level_sampling()
    test_vs_sequence_sampling()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 所有测试通过！帧级别采样逻辑正确")
    else:
        print("❌ 测试失败！需要检查实现")
    print("=" * 70)