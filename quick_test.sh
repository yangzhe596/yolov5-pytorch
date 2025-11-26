#!/bin/bash
# 快速验证训练脚本功能
# 
# 使用方法：
#   ./quick_test.sh [模态]
#
# 示例：
#   ./quick_test.sh rgb    # 快速验证 RGB 模态训练
#   ./quick_test.sh event  # 快速验证 Event 模态训练
#   ./quick_test.sh        # 默认验证 RGB 模态

# 默认参数
MODALITY=${1:-rgb}

# Python 路径
PYTHON="/home/yz/.conda/envs/torch/bin/python3"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}⚡ 快速验证训练脚本${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "模态: ${GREEN}${MODALITY}${NC}"
echo -e "模式: ${YELLOW}快速验证（2 epoch × 100 batch）${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

echo -e "${YELLOW}说明：${NC}"
echo "  - 仅运行 2 个 epoch"
echo "  - 每个 epoch 最多 100 个 batch"
echo "  - 用于快速验证训练流程是否正确"
echo "  - 预计耗时：约 5-10 分钟"
echo ""

# 执行
echo "执行命令: $PYTHON train_fred.py --modality $MODALITY --quick_test"
echo ""
$PYTHON train_fred.py --modality $MODALITY --quick_test

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 快速验证完成！${NC}"
    echo ""
    echo "训练流程验证通过，可以开始完整训练："
    echo -e "  ${GREEN}$PYTHON train_fred.py --modality $MODALITY${NC}"
else
    echo ""
    echo -e "${RED}✗ 快速验证失败${NC}"
    echo "请检查错误信息并修复问题"
    exit 1
fi
