"""
快速验证解冻训练修复
"""
import sys
import os


def check_train_fusion_fix():
    """检查 train_fred_fusion.py 是否已修复"""
    
    train_file = '/mnt/data/code/yolov5-pytorch/train_fred_fusion.py'
    
    if not os.path.exists(train_file):
        print("❌ 文件不存在:", train_file)
        return False
    
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = []
    
    # 检查 1: 解冻阶段重新创建 scaler
    if 'scaler = torch.cuda.amp.GradScaler(enabled=fp16)' in content:
        # 确保在解冻训练部分
        if '第二阶段：解冻训练' in content and \
           content.find('第二阶段：解冻训练') < content.find('scaler = torch.cuda.amp.GradScaler(enabled=fp16)'):
            checks.append(("✓", "解冻训练阶段重新创建 scaler"))
        else:
            checks.append(("⚠", "找到 scaler 创建代码，但位置可能不对"))
    else:
        checks.append(("✗", "未找到解冻阶段的 scaler 创建"))
    
    # 检查 2: 正确的混合精度判断
    if 'use_autocast = fp16 and (scaler is not None)' in content:
        checks.append(("✓", "修复了混合精度判断条件"))
    else:
        checks.append(("✗", "未修复混合精度判断条件"))
    
    # 检查 3: 检查点保存
    if "freeze_epoch_{epoch+1}_weights.pth" in content:
        checks.append(("✓", "添加了检查点自动保存功能"))
    else:
        checks.append(("✗", "未添加检查点保存功能"))
    
    # 检查 4: 检查点加载（解冻阶段）
    section_start = content.find("第二阶段：解冻训练")
    section_end = content.find("# === 仅评估模式 ===")
    if section_start > 0 and section_end > section_start:
        unfreeze_section = content[section_start:section_end]
        if 'freeze_last_epoch_weights.pth' in unfreeze_section:
            checks.append(("✓", "添加了冻结训练检查点加载"))
        else:
            checks.append(("⚠", "未添加冻结训练检查点加载"))
    
    # 检查 5: 无 scaler=None 的调用
    if 'fit_one_epoch_fusion(..., scaler=None, ...)' in content or \
       ', None, save_period' in content:
        checks.append(("✗", "仍存在 scaler=None 的调用"))
    else:
        checks.append(("✓", "移除了 scaler=None 的错误调用"))
    
    return checks


def main():
    print("="*60)
    print("解冻训练停止问题修复验证")
    print("="*60)
    
    checks = check_train_fusion_fix()
    
    if checks is False:
        print("\n⚠️ 无法检查文件")
        return
    
    print("\n修复检查结果:")
    print("-" * 60)
    
    all_ok = True
    for status, message in checks:
        print(f"{status} {message}")
        if status == '✗':
            all_ok = False
    
    print("-" * 60)
    
    if all_ok:
        print("\n🎉 所有修复已完成！")
        print("\n现在可以正常训练:")
        print("  python train_fred_fusion.py --modality rgb")
        print("\n如果训练中断，可恢复:")
        print("  ./resume_training.sh rgb")
    else:
        print("\n⚠️ 部分修复尚未应用")
        print("\n建议:")
        print("  1. 确认使用了修复后的 train_fred_fusion.py")
        print("  2. 查看 解冻训练停止问题修复.md 了解详情")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()