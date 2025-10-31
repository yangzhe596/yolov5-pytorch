#!/bin/bash
# FRED 数据集快速可视化脚本
# 
# 使用方法：
#   ./quick_visualize.sh [序列ID] [模态] [最大帧数]
#
# 示例：
#   ./quick_visualize.sh 0 rgb 100    # 可视化序列0的前100帧（RGB）
#   ./quick_visualize.sh 1 event 200  # 可视化序列1的前200帧（Event）
#   ./quick_visualize.sh 0 rgb        # 可视化序列0的所有帧（RGB）

# 默认参数
SEQUENCE=${1:-0}
MODALITY=${2:-rgb}
MAX_FRAMES=${3:-}

# Python 路径
PYTHON="/home/yz/miniforge3/envs/torch/bin/python3"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}FRED 快速可视化${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "序列: ${GREEN}${SEQUENCE}${NC}"
echo -e "模态: ${GREEN}${MODALITY}${NC}"
if [ -n "$MAX_FRAMES" ]; then
    echo -e "最大帧数: ${GREEN}${MAX_FRAMES}${NC}"
else
    echo -e "最大帧数: ${GREEN}全部${NC}"
fi
echo -e "${BLUE}======================================${NC}"
echo ""

# 构建命令
CMD="$PYTHON visualize_fred_sequences.py --modality $MODALITY --sequence $SEQUENCE --export-video --no-window"

if [ -n "$MAX_FRAMES" ]; then
    CMD="$CMD --max-frames $MAX_FRAMES"
fi

# 执行
echo "执行命令: $CMD"
echo ""
$CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 可视化完成！${NC}"
    echo ""
    echo "视频文件位置："
    ls -lh visualizations/sequence_${SEQUENCE}_${MODALITY}.mp4 2>/dev/null || echo "未找到视频文件"
else
    echo ""
    echo -e "${RED}✗ 可视化失败${NC}"
    exit 1
fi
