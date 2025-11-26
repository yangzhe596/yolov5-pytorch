#!/usr/bin/env python3
"""
测试 time_diff 计算修复
验证修复后的 time_diff 不再是 Infinity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convert_fred_to_fusion import FREDFusionConverterV2
import json

def test_time_diff_fix():
    """测试 time_diff 计算修复"""
    print("=" * 70)
    print("测试 time_diff 计算修复")
    print("=" * 70)
    
    # 创建转换器
    converter = FREDFusionConverterV2(
        fred_root="/mnt/data/datasets/fred",
        output_root="/mnt/data/code/yolov5-pytorch/datasets/fred_fusion_test_fix",
        split_mode="frame",
        simple=True,
        simple_ratio=0.05  # 5% 的数据进行快速测试
    )
    
    # 获取前几个序列进行测试
    sequences = converter.get_all_sequences()[:3]
    print(f"使用 {len(sequences)} 个序列进行测试")
    
    # 处理一个序列并检查配对结果
    seq_id = sequences[0]
    print(f"\n处理序列 {seq_id}...")
    
    annotations_dict, paired_frames, images = converter.process_sequence(seq_id)
    
    # 统计不同状态的帧
    status_counts = {}
    time_diff_values = []
    
    for frame_info in paired_frames:
        status = frame_info['status']
        time_diff = frame_info['time_diff']
        
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
        
        if time_diff != float('inf'):
            time_diff_values.append(time_diff)
    
    print(f"\n配对结果统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} 帧")
    
    print(f"\ntime_diff 统计:")
    print(f"  非 Infinity 值数量: {len(time_diff_values)}")
    if time_diff_values:
        print(f"  最小值: {min(time_diff_values):.3f} ms")
        print(f"  最大值: {max(time_diff_values):.3f} ms")
        print(f"  平均值: {sum(time_diff_values)/len(time_diff_values):.3f} ms")
    
    # 检查是否有 Infinity 值
    infinity_count = sum(1 for frame_info in paired_frames if frame_info['time_diff'] == float('inf'))
    total_frames = len(paired_frames)
    
    print(f"\nInfinity 检查:")
    print(f"  Infinity 值数量: {infinity_count}/{total_frames} ({infinity_count/total_frames:.1%})")
    
    if infinity_count == 0:
        print("✅ 没有 Infinity 值，修复成功！")
        return True
    else:
        print("⚠️  仍有 Infinity 值，需要进一步检查")
        
        # 检查哪些状态有 Infinity
        infinity_by_status = {}
        for frame_info in paired_frames:
            if frame_info['time_diff'] == float('inf'):
                status = frame_info['status']
                if status not in infinity_by_status:
                    infinity_by_status[status] = 0
                infinity_by_status[status] += 1
        
        print("  Infinity 值分布:")
        for status, count in infinity_by_status.items():
            print(f"    {status}: {count}")
        
        return False

def test_small_conversion():
    """测试小规模转换"""
    print("\n" + "=" * 70)
    print("测试小规模数据转换")
    print("=" * 70)
    
    # 创建转换器
    converter = FREDFusionConverterV2(
        fred_root="/mnt/data/datasets/fred",
        output_root="/mnt/data/code/yolov5-pytorch/datasets/fred_fusion_test_fix",
        split_mode="frame",
        simple=True,
        simple_ratio=0.02  # 2% 的数据进行快速测试
    )
    
    # 运行转换（只处理训练集）
    try:
        converter.generate_all_fusion(splits=['train'])
        
        # 检查生成的文件
        train_file = converter.output_root / "annotations" / "instances_train.json"
        
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
            
            # 统计 time_diff 值
            infinity_count = 0
            total_count = 0
            time_diff_values = []
            
            for image in data.get('images', []):
                if 'time_diff' in image:
                    total_count += 1
                    if image['time_diff'] == float('inf') or image['time_diff'] == 'Infinity':
                        infinity_count += 1
                    else:
                        time_diff_values.append(image['time_diff'])
            
            print(f"\n生成的标注文件统计:")
            print(f"  总图像数: {total_count}")
            print(f"  Infinity 值数量: {infinity_count} ({infinity_count/total_count:.1%})")
            
            if time_diff_values:
                print(f"  有效 time_diff 范围: {min(time_diff_values):.3f} - {max(time_diff_values):.3f} ms")
                print(f"  平均 time_diff: {sum(time_diff_values)/len(time_diff_values):.3f} ms")
            
            return infinity_count == 0
        else:
            print("❌ 标注文件未生成")
            return False
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

if __name__ == "__main__":
    print("测试 time_diff 计算修复")
    print("验证修复后的 time_diff 不再是 Infinity")
    print()
    
    # 运行测试
    test1_passed = test_time_diff_fix()
    test2_passed = test_small_conversion()
    
    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("✅ 所有测试通过！time_diff 修复成功")
    else:
        print("❌ 部分测试失败，需要进一步检查")
    print("=" * 70)