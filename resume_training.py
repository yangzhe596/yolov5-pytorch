"""
Fusion 模型断点续练工具

用于从检查点恢复训练，避免因崩溃或停止导致的损失
"""
import os
import torch
import argparse
from train_fred_fusion import main as train_main


def find_latest_checkpoint(log_dir, phase="unfreeze"):
    """
    查找最新的检查点
    
    Args:
        log_dir: 日志目录
        phase: 训练阶段 ("freeze" 或 "unfreeze")
        
    Returns:
        检查点路径或 None
    """
    import re
    
    checkpoint_files = []
    for file in os.listdir(log_dir):
        if file.endswith('.pth') and phase in file:
            # 提取 epoch 数字
            match = re.search(rf'{phase}_epoch_(\d+)_weights\.pth', file)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, os.path.join(log_dir, file)))
    
    if not checkpoint_files:
        return None
    
    # 按 epoch 降序排序，返回最新的
    checkpoint_files.sort(reverse=True, key=lambda x: x[0])
    return checkpoint_files[0]  # (epoch, path)


def resume_training_from_checkpoint(args, checkpoint_path):
    """
    从检查点恢复训练
    
    Args:
        args: 命令行参数
        checkpoint_path: 检查点路径
        
    Returns:
        updated_args: 更新后的参数
    """
    print("="*60)
    print(f"从检查点恢复训练")
    print("="*60)
    print(f"检查点: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    
    print(f"恢复到 epoch: {start_epoch}")
    
    # 更新参数
    args.resuming = True
    args.checkpoint_path = checkpoint_path
    
    return args, start_epoch


def main():
    """
    主函数：提供命令行接口用于断点续练
    """
    parser = argparse.ArgumentParser(description="Fusion 模型断点续练")
    
    # 基本参数
    parser.add_argument('--modality', type=str, default='rgb', 
                       choices=['rgb', 'event'], help='训练模态')
    parser.add_argument('--log_dir', type=str, default=None, 
                       help='日志目录路径（如果不指定，自动查找）')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='手动指定检查点路径')
    
    # Fusion 特定参数（透传给 train_fred_fusion.py）
    mode_group = parser.add_argument_group('训练模式')
    mode_group.add_argument('--freeze_training', action='store_true',
                          help='仅进行冻结训练，不进行解冻训练')
    mode_group.add_argument('--no_eval', action='store_true',
                          help='禁用评估')
    mode_group.add_argument('--no_eval_map', action='store_true',
                          help='禁用 mAP 评估')
    mode_group.add_argument('--quick_test', action='store_true',
                          help='快速测试模式')
    
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--backbone', type=str, default='cspdarknet',
                           choices=['cspdarknet', 'convnext_tiny', 'convnext_small', 'swin_transformer_tiny'],
                           help='主干网络')
    model_group.add_argument('--phi', type=str, default='s',
                           choices=['s', 'm', 'l', 'x'],
                           help='YOLOv5 版本')
    model_group.add_argument('--resume_last', action='store_true',
                           help='从最后保存的模型恢复（不支持 checkpoint）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 确定日志目录
    if args.log_dir is None:
        if args.modality == 'rgb':
            log_dir = 'logs/fred_rgb'
        else:
            log_dir = 'logs/fred_event'
    else:
        log_dir = args.log_dir
    
    print(f"\n日志目录: {log_dir}")
    
    # 查找检查点
    checkpoint_path = None
    if args.resume_checkpoint:
        checkpoint_path = args.resume_checkpoint
        phase = "freeze" if "freeze" in checkpoint_path else "unfreeze"
    else:
        # 自动查找最新检查点（优先查找 unfreeze）
        checkpoint_path = find_latest_checkpoint(log_dir, "unfreeze")
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(log_dir, "freeze")
        
        if checkpoint_path:
            phase = "freeze" if "freeze" in checkpoint_path[1] else "unfreeze"
            checkpoint_path = checkpoint_path[1]
    
    if checkpoint_path is None:
        print("❌ 未找到检查点文件")
        print(f"检查目录: {log_dir}")
        return
    
    # 更新参数以支持断点续练
    args.resuming = True
    args.checkpoint_path = checkpoint_path
    args.resume_last = False  # 不使用最后保存的模型，而是使用 checkpoint
    
    print(f"✓ 使用检查点: {checkpoint_path}")
    print(f"✓ 训练阶段: {'冻结训练' if 'freeze' in checkpoint_path else '解冻训练'}")
    
    # 加载检查点以获取起始 epoch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    resume_epoch = checkpoint['epoch']
    print(f"✓ 从 epoch {resume_epoch} 恢复训练")
    
    # 显示恢复信息
    print("\n" + "="*60)
    print("断点续练配置")
    print("="*60)
    print(f"检查点: {os.path.basename(checkpoint_path)}")
    print(f"起始 epoch: {resume_epoch}")
    print(f"训练模态: {args.modality.upper()}")
    if args.freeze_training:
        print(f"训练模式: 冻结训练（仅冻结阶段）")
    else:
        print(f"训练模式: 冻结 + 解冻训练")
    
    # 调用主训练函数
    # 注意：需要修改 train_fred_fusion.py 以支持 checkpoint 参数
    print("\n⚠️  注意: 这只是一个示例实现")
    print("实际使用时需要:")
    print("1. 修改 train_fred_fusion.py 的参数解析")
    print("2. 添加 --resume_checkpoint 参数支持")
    print("3. 在训练开始时加载 checkpoint")
    
    # 构建参数列表
    argv = [
        '--modality', args.modality,
        '--resume_checkpoint', checkpoint_path
    ]
    
    if args.freeze_training:
        argv.append('--freeze_training')
    if args.no_eval:
        argv.append('--no_eval')
    if args.no_eval_map:
        argv.append('--no_eval_map')
    if args.quick_test:
        argv.append('--quick_test')
    
    return


def simple_resume_example():
    """
    简单的断点续练示例
    """
    print("\n" + "="*60)
    print("简单断点续练方法")
    print("="*60)
    print("\n方法 1: 手动修改模型路径（推荐）")
    print("  step 1: 打开 train_fred_fusion.py")
    print("  step 2: 找到 model_path 设置")
    print("  step 3: 设置为检查点路径:")
    print("    model_path = 'logs/fred_fusion/freeze_epoch_50_weights.pth'")
    print("  step 4: 设置 Init_Epoch:")
    print("    Init_Epoch = 50")
    print("  step 5: 运行训练:")
    print("    python train_fred_fusion.py --modality rgb")
    
    print("\n方法 2: 使用 --resume_last（需先保存最后模型）")
    print("  python train_fred_fusion.py --modality rgb --resume_last")
    
    print("\n方法 3: 批量恢复脚本")
    print("  bash resume_training.sh")


def check_cuda_memory():
    """
    检查 CUDA 内存状态
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            # 清理显存
            torch.cuda.empty_cache()
            
            # 获取显存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            
            print("\n" + "="*60)
            print("CUDA 显存状态")
            print("="*60)
            print(f"总显存: {total_memory:.2f} GB")
            print(f"已分配: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
            print(f"已保留: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
            print(f"可用显存: {total_memory - reserved:.2f} GB")
            
            return total_memory > 8  # 至少需要 8GB 显存
        except:
            return True
    else:
        print("⚠️  未检测到 CUDA")
        return False


if __name__ == "__main__":
    # 检查显存
    if not check_cuda_memory():
        print("\n🚨 警告: 显存不足，最大可能会影响训练稳定性")
        print("建议:")
        print("  - 减小 batch size")
        print("  - 减小 input_shape")
        print("  - 使用 --no_eval_map 禁用 mAP 评估")
    
    # 显示使用说明
    simple_resume_example()
    
    # 提示用户如何实际使用
    print("\n" + "="*60)
    print("直接恢复训练的方法")
    print("="*60)
    print("\n在 train_fred_fusion.py 中添加以下代码:")
    print("""
# 加载检查点
if args.checkpoint_path:
    print(f"加载检查点: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model_train.load_state_dict(checkpoint['model'])
    if ema:
        ema.ema.load_state_dict(checkpoint['ema'])
    print(f"检查点加载成功")
    
    # 获取恢复 epoch
    Init_Epoch = checkpoint['epoch']
    UnFreeze_Epoch = max(UnFreeze_Epoch, Init_Epoch + 1)
""")