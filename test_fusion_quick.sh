#!/bin/bash
# 快速测试融合脚本 - 实时输出版本

echo "=========================================="
echo "FRED 融合数据集 - 快速测试 (实时输出)"
echo "=========================================="

# 简化模式测试
echo ""
echo "开始转换（简化模式）..."
echo "预计时间: 2-3 分钟"
echo "=========================================="

/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion.py \
    --fred-root /mnt/data/datasets/fred \
    --output-root datasets/fred_fusion_test \
    --split-mode frame \
    --threshold 0.033 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --simple

# 检查结果
echo ""
echo "=========================================="
echo "转换完成，检查输出..."
echo "=========================================="
ls -lh datasets/fred_fusion_test/

echo ""
echo "标注文件:"
ls -lh datasets/fred_fusion_test/annotations/

echo ""
echo "融合信息:"
cat datasets/fred_fusion_test/fusion_info.json | python3 -m json.tool

echo ""
echo "=========================================="
echo "快速测试完成！"
echo "=========================================="