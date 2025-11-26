#!/bin/bash
echo "开始执行 Fusion 训练脚本..."
echo "Python 路径: $1"
echo "参数: $@"
echo "当前目录: $(pwd)"
echo "时间: $(date)"

# 执行脚本
"$1" train_fred_fusion.py --compression_ratio 1.0 --modality dual

echo "脚本执行完成，退出码: $?"