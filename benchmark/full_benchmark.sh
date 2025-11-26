#!/bin/bash
# 完整基准测试脚本 - 测试不同配置的性能

PYTHON=/home/yz/.conda/envs/torch/bin/python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "YOLOv5 完整性能基准测试"
echo "========================================================================"
echo ""

# 测试 1: 默认配置 (RGB)
echo "========================================================================"
echo "测试 1: RGB 模态 - 默认配置"
echo "========================================================================"
$PYTHON "$SCRIPT_DIR/training_benchmark.py" \
    --modality rgb \
    --num_batches 50 \
    --batch_size 8

echo ""
echo "========================================================================"
echo "测试 2: RGB 模态 - 增加 workers"
echo "========================================================================"
$PYTHON "$SCRIPT_DIR/training_benchmark.py" \
    --modality rgb \
    --num_batches 50 \
    --batch_size 8 \
    --num_workers 16

echo ""
echo "========================================================================"
echo "测试 3: RGB 模态 - 增加 prefetch_factor"
echo "========================================================================"
$PYTHON "$SCRIPT_DIR/training_benchmark.py" \
    --modality rgb \
    --num_batches 50 \
    --batch_size 8 \
    --prefetch_factor 8

echo ""
echo "========================================================================"
echo "测试 4: RGB 模态 - 增加 batch_size"
echo "========================================================================"
$PYTHON "$SCRIPT_DIR/training_benchmark.py" \
    --modality rgb \
    --num_batches 50 \
    --batch_size 16

echo ""
echo "========================================================================"
echo "所有测试完成"
echo "========================================================================"
