#!/bin/bash
# simple 模式快速测试脚本

echo "======================================================================"
echo "FRED 融合数据集 - simple 模式快速测试"
echo "======================================================================"
echo ""

# 默认参数
OUTPUT_DIR="datasets/fred_fusion_v2_simple"
SIMPLE_RATIO=0.1
TRAIN_RATIO=1.0
VAL_RATIO=0
TEST_RATIO=0

# 帮助信息
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --ratio RATIO    采样比例 (默认: 0.1)"
    echo "  -o, --output DIR     输出目录 (默认: ${OUTPUT_DIR})"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  bash quick_simple_test.sh                    # 默认 (10%)"
    echo "  bash quick_simple_test.sh -r 0.05            # 5% 数据"
    echo "  bash quick_simple_test.sh -r 0.02 -o output  # 2% 数据"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ratio)
            SIMPLE_RATIO="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

echo "配置参数:"
echo "  采样比例: $SIMPLE_RATIO"
echo "  输出目录: $OUTPUT_DIR"
echo "  划分模式: frame (simplified)"
echo ""

# 运行转换
echo "开始转换..."
echo ""

/home/yz/.conda/envs/torch/bin/python3 convert_fred_to_fusion_v2.py \
    --split-mode frame \
    --simple \
    --simple-ratio $SIMPLE_RATIO \
    --train-ratio $TRAIN_RATIO \
    --val-ratio $VAL_RATIO \
    --test-ratio $TEST_RATIO

# 检查结果
echo ""
if [ $? -eq 0 ]; then
    echo "✅ 转换完成！"
    echo ""
    echo "输出文件:"
    ls -lh ${OUTPUT_DIR}/annotations/ 2>/dev/null || echo "  未找到输出文件"
    echo ""
    echo "可通过以下命令查看数据集信息:"
    echo "  python verify_coco_dataset.py --json ${OUTPUT_DIR}/annotations/instances_train.json"
else
    echo "❌ 转换失败！"
    exit 1
fi