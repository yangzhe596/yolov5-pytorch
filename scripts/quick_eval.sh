#!/bin/bash
# 快速评估已训练的FRED模型

echo "=========================================="
echo "FRED模型快速评估脚本"
echo "=========================================="
echo ""

# 检查参数
if [ "$1" == "rgb" ] || [ "$1" == "event" ]; then
    MODALITY=$1
else
    echo "使用方法: ./quick_eval.sh [rgb|event]"
    echo "示例: ./quick_eval.sh rgb"
    echo ""
    echo "默认评估RGB模型..."
    MODALITY="rgb"
fi

# 设置路径
MODEL_DIR="logs/fred_${MODALITY}"
BEST_MODEL="${MODEL_DIR}/best_epoch_weights.pth"

# 检查模型是否存在
if [ ! -f "$BEST_MODEL" ]; then
    echo "❌ 错误: 找不到模型文件 $BEST_MODEL"
    echo ""
    echo "可用的模型文件:"
    ls -lh ${MODEL_DIR}/*.pth 2>/dev/null || echo "  (无)"
    exit 1
fi

echo "模型路径: $BEST_MODEL"
echo "模态: ${MODALITY^^}"
echo ""
echo "开始评估..."
echo "=========================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 运行评估
$PYTHON "$SCRIPT_DIR/eval_fred.py" \
    --model_path "$BEST_MODEL" \
    --modality "$MODALITY" \
    --confidence 0.05 \
    --nms_iou 0.5

echo ""
echo "=========================================="
echo "评估完成！"
echo ""
echo "结果文件:"
echo "  - ${MODEL_DIR}/eval_result_${MODALITY}.txt"
echo "  - map_out/ (详细结果)"
echo "=========================================="
