#!/bin/bash
# 脚本功能说明
# 快速测试融合转换脚本（使用简化模式）
# 
# 使用方法:
#   bash quick_test_fusion.sh
#   bash quick_test_fusion.sh --simple
#   bash quick_test_fusion.sh --threshold 0.02

set -e  # 遇到错误立即退出

echo "=========================================="
echo "FRED 融合数据集 - 快速测试"
echo "=========================================="

# 默认参数
THRESHOLD=0.033  # 33ms
MODE="frame"
SIMPLE_FLAG=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            SIMPLE_FLAG="--simple"
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "可用参数:"
            echo "  --simple              启用简化模式（不会处理全部数据）"
            echo "  --threshold <值>      设置时间容差阈值（秒，默认 0.033）"
            echo "  --mode <frame|sequence> 设置划分模式（默认 frame）"
            exit 1
            ;;
    esac
done

echo "参数配置:"
echo "  时间容差: ${THRESHOLD}s"
echo "  划分模式: ${MODE}"
if [ -n "$SIMPLE_FLAG" ]; then
    echo "  简化模式: 启用"
else
    echo "  简化模式: 禁用"
fi
echo ""

# 检查 Python 环境
PYTHON_PATH="/home/yz/.conda/envs/torch/bin/python3"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "错误: Python 环境未找到: $PYTHON_PATH"
    exit 1
fi

# 检查脚本是否存在
SCRIPT_PATH="convert_fred_to_fusion.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 脚本未找到: $SCRIPT_PATH"
    exit 1
fi

# 执行转换
echo "开始转换..."
echo "=========================================="

CMD="$PYTHON_PATH $SCRIPT_PATH \
    --fred-root /mnt/data/datasets/fred \
    --output-root datasets/fred_fusion_test \
    --split-mode $MODE \
    --threshold $THRESHOLD \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    $SIMPLE_FLAG"

echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD

# 检查输出
OUTPUT_DIR="datasets/fred_fusion_test"
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "✅ 转换完成！"
    echo "=========================================="
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    echo "目录结构:"
    tree -L 2 "$OUTPUT_DIR" 2>/dev/null || ls -R "$OUTPUT_DIR"
    
    echo ""
    echo "标注文件:"
    find "$OUTPUT_DIR" -name "*.json" -type f
    
    echo ""
    echo "检查融合信息..."
    if [ -f "$OUTPUT_DIR/fusion_info.json" ]; then
        echo "✅ fusion_info.json 生成成功"
        cat "$OUTPUT_DIR/fusion_info.json" | python3 -m json.tool | head -20
    fi
    
    echo ""
    echo "检查数据统计..."
    for SPLIT in train val test; do
        ANNOT_FILE="$OUTPUT_DIR/annotations/instances_${SPLIT}.json"
        if [ -f "$ANNOT_FILE" ]; then
            echo "----------------------------------------"
            echo "$SPLIT 划分:"
            python3 -c "
import json
with open('$ANNOT_FILE', 'r') as f:
    data = json.load(f)
print(f'  图像数: {len(data[\\\"images\\\"])}')
print(f'  标注数: {len(data[\\\"annotations\\\"])}')
if 'info' in data:
    print(f'  数据描述: {data[\\\"info\\\"].get(\\\"description\\\", \\\"N/A\\\")}')
# 统计融合状态
statuses = {}
for img in data.get('images', []):
    status = img.get('fusion_status', 'unknown')
    statuses[status] = statuses.get(status, 0) + 1
if statuses:
    print(f'  融合状态分布:')
    for status, count in sorted(statuses.items()):
        pct = count / len(data['images']) * 100 if data['images'] else 0
        print(f'    - {status}: {count} ({pct:.1f}%)')
"
        fi
    done
else
    echo "❌ 输出目录不存在: $OUTPUT_DIR"
    exit 1
fi