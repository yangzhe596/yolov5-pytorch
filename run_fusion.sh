#!/bin/bash
echo "执行 Fusion 训练脚本..."
# 使用正确的 Python 环境和参数
exec /home/yz/.conda/envs/torch/bin/python3 train_fred_fusion.py "$@"