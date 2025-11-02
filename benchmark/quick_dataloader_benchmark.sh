#!/bin/bash
# 快速数据加载基准测试脚本

PYTHON=/home/yz/miniforge3/envs/torch/bin/python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "数据加载性能详细分析"
echo "========================================================================"
echo ""

# 测试 RGB 模态
echo "测试 RGB 模态 (100 samples)..."
$PYTHON "$SCRIPT_DIR/dataloader_benchmark.py" \
    --modality rgb \
    --num_samples 100

echo ""
echo "========================================================================"
echo "测试完成"
echo "========================================================================"
