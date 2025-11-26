#!/bin/bash
# 快速基准测试脚本

PYTHON=/home/yz/.conda/envs/torch/bin/python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "YOLOv5 快速性能基准测试"
echo "========================================================================"
echo ""

# 测试 RGB 模态
echo "测试 RGB 模态 (50 batches)..."
$PYTHON "$SCRIPT_DIR/training_benchmark.py" \
    --modality rgb \
    --num_batches 50 \
    --batch_size 8

echo ""
echo "========================================================================"
echo "测试完成"
echo "========================================================================"
