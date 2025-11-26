#!/bin/bash
# Fusion 模型训练恢复脚本
# 用于在训练停止时快速恢复

set -e

echo "============================================================"
echo "Fusion 模型训练恢复"
echo "============================================================"

# 参数配置
MODALITY=${1:-rgb}  # 默认 rgb 模态
LOG_DIR="logs/fred_${MODALITY}"
RESUME_MODE=${2:-auto}  # auto | freeze | unfreeze

echo "训练模态: $MODALITY"
echo "日志目录: $LOG_DIR"
echo "恢复模式: $RESUME_MODE"

# 检查目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 错误: 日志目录不存在 - $LOG_DIR"
    echo "请先运行训练脚本创建日志目录"
    exit 1
fi

# 查找检查点
find_checkpoint() {
    local phase=$1
    echo "查找 ${phase} 检查点..."
    
    # 使用 Python 查找最新的检查点
    local checkpoint=$($HOME/.conda/envs/torch/bin/python -c "
import os
import re
import sys

log_dir = '$LOG_DIR'
phase = '$phase'

checkpoint_files = []
for file in os.listdir(log_dir):
    if file.endswith('.pth') and phase in file:
        match = re.search(rf'{phase}_epoch_(\d+)_weights\.pth', file)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((epoch, os.path.join(log_dir, file)))

if checkpoint_files:
    checkpoint_files.sort(reverse=True, key=lambda x: x[0])
    print(checkpoint_files[0][1])  # 只打印路径
else:
    sys.exit(1)
")
    
    echo "$checkpoint"
}

# 自动选择检查点
CHECKPOINT=""
if [ "$RESUME_MODE" = "auto" ]; then
    # 优先查找 unfreeze
    CHECKPOINT=$(find_checkpoint "unfreeze" 2>/dev/null || true)
    
    if [ -z "$CHECKPOINT" ]; then
        # 没有 unfreeze，查找 freeze
        CHECKPOINT=$(find_checkpoint "freeze" 2>/dev/null || true)
    fi
elif [ "$RESUME_MODE" = "freeze" ]; then
    CHECKPOINT=$(find_checkpoint "freeze")
elif [ "$RESUME_MODE" = "unfreeze" ]; then
    CHECKPOINT=$(find_checkpoint "unfreeze")
fi

if [ -z "$CHECKPOINT" ]; then
    echo "❌ 未找到检查点文件"
    echo "搜索目录: $LOG_DIR"
    echo ""
    echo "可用的检查点文件:"
    ls -1 "$LOG_DIR"/*.pth 2>/dev/null || echo "无检查点文件"
    exit 1
fi

echo "✓ 找到检查点: $CHECKPOINT"

# 提取 epoch 信息
EPOCH=$(echo "$CHECKPOINT" | grep -oP 'epoch_\K\d+' || echo "unknown")
PHASE=""
if echo "$CHECKPOINT" | grep -q "freeze"; then
    PHASE="freeze"
fi

echo "✓ 恢复 epoch: $EPOCH"
echo "✓ 训练阶段: $PHASE"

# 验证检查点文件
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 检查点文件不存在: $CHECKPOINT"
    exit 1
fi

# 显示配置
echo ""
echo "============================================================"
echo "恢复训练配置"
echo "============================================================"
echo "检查点文件: $(basename "$CHECKPOINT")"
echo "恢复 epoch: $EPOCH"
echo ""
echo "即将执行的命令:"
echo "python train_fred_fusion.py --modality $MODALITY \\"
echo "  --resume_checkpoint $CHECKPOINT"

# 确认用户
read -p "确认开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消"
    exit 0
fi

echo ""
echo "============================================================"
echo "开始恢复训练"
echo "============================================================"
echo ""

# 执行训练
# 注意：这需要 train_fred_fusion.py 支持 --resume_checkpoint 参数
# 如果不支持，可以手动修改

$HOME/.conda/envs/torch/bin/python train_fred_fusion.py \
    --modality $MODALITY \
    --resume_checkpoint "$CHECKPOINT"

echo ""
echo "============================================================"
echo "训练完成"
echo "============================================================"