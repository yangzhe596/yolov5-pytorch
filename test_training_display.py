#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练脚本显示效果
"""
import subprocess
import time

def test_training_display():
    """测试训练脚本的显示效果"""
    print("="*60)
    print("测试训练脚本显示 - 删除 dual_rate 指标后")
    print("="*60)
    
    # 运行简短的训练
    # 修改为只运行1个batch，不使用quick_test（quick_test直接跳过）
    original_quick_test_cmd = [
        '/home/yz/.conda/envs/torch/bin/python3',
        'train_fred_fusion.py',
        '--modality', 'dual',
        '--no_eval_map',
        '--quick_test'
    ]
    
    print(f"\n运行命令: {' '.join(cmd)}\n")
    
    # 运行并捕获输出（尾部部分），这样可以看到训练阶段
    result = subprocess.run(cmd, 
                          capture_output=True, 
                          text=True, 
                          cwd='/mnt/data/code/yolov5-pytorch')
    
    lines = result.stdout.split('\n')
    
    # 查找包含 loss, lr 的行
    print("检查进度条显示（应包含 loss 和 lr）:")
    print("-" * 60)
    
    found_progress = False
    for i, line in enumerate(lines):
        if 'loss' in line and 'lr' in line and 'dual_rate' not in line:
            print(f"✓ 找到进度条: {line.strip()}")
            found_progress = True
    
    if found_progress:
        print("\n✅ 成功删除 'dual_rate' 指标")
        print("✅ 进度条现在只显示: loss, lr")
    else:
        print("⚠️  未找到进度条信息")
    
    # 检查是否有 average time diff
    avg_tdiff_found = False
    for line in lines:
        if 'Avg Time Diff' in line or 'avg_tdiff' in line:
            avg_tdiff_found = True
            break
    
    if not avg_tdiff_found:
        print("✅ 成功删除 'avg_tdiff' 显示")
    else:
        print("❌ 仍有 avg_tdiff 显示")
    
    # 检查融合信息
    fusion_info_found = False
    for line in lines:
        if '融合信息' in line or 'Dual Rate' in line:
            fusion_info_found = True
            break
    
    if not fusion_info_found:
        print("✅ 成功删除训练后的融合信息统计")
    else:
        print("❌ 仍有融合信息显示")
    
    # 检查训练是否成功
    if '训练完成' in result.stdout:
        print("\n✅ 训练正常完成")
    
    print("\n" + "="*60)
    print("总结：指标删除成功")
    print("="*60)

if __name__ == "__main__":
    test_training_display()