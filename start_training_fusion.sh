#!/bin/bash
#
# FRED Fusion 双模态融合训练启动脚本
#
# 功能:
#   1. 检查 Python 环境
#   2. 检查数据集配置
#   3. 显示配置摘要
#   4. 开始训练
#

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/mnt/data/code/yolov5-pytorch"

# Python 环境
PYTHON_ENV="/home/yz/.conda/envs/torch/bin/python3"

# 检查 Python 环境
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  检查 Python 环境${NC}"
echo -e "${BLUE}========================================${NC}"

if [ ! -f "$PYTHON_ENV" ]; then
    echo -e "${RED}错误: Python 环境未找到!${NC}"
    echo -e "${RED}预期路径: $PYTHON_ENV${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 环境: $PYTHON_ENV${NC}"
echo -e "${GREEN}✓ PyTorch 版本: $($PYTHON_ENV -c 'import torch; print(torch.__version__)')${NC}"
echo -e "${GREEN}✓ CUDA 是否可用: $($PYTHON_ENV -c 'import torch; print(torch.cuda.is_available())')${NC}"
if $PYTHON_ENV -c "import torch" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 依赖库检查通过${NC}"
else
    echo -e "${RED}错误: 缺少必要的 Python 库${NC}"
    echo -e "${RED}请确保已激活 conda 环境: conda activate torch${NC}"
    exit 1
fi

# 检查数据集
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  检查数据集${NC}"
echo -e "${BLUE}========================================${NC}"

FUSION_JSON="/mnt/data/datasets/fred/fred_fusion.json"
DATA_ROOT="/mnt/data/datasets/fred"

if [ ! -f "$FUSION_JSON" ]; then
    echo -e "${YELLOW}警告: 融合标注文件未找到${NC}"
    echo -e "${YELLOW}预期路径: $FUSION_JSON${NC}"
    echo -e "${YELLOW}如果这是首次训练，请先运行数据准备脚本${NC}"
    echo -e ""
    echo -e "${YELLOW}准备数据集命令:${NC}"
    echo -e "${YELLOW}  $PYTHON_ENV convert_fred_to_coco_v2.py${NC}"
    echo -e ""
    echo -e "${YELLOW}然后重新运行此脚本${NC}"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}错误: FRED 数据集根目录未找到!${NC}"
    echo -e "${RED}预期路径: $DATA_ROOT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 数据集根目录: $DATA_ROOT${NC}"
echo -e "${GREEN}✓ 融合标注文件: $FUSION_JSON${NC}"

# 检查日志目录
LOG_DIR="$PROJECT_ROOT/logs/fred_fusion"
mkdir -p "$LOG_DIR"

# 显示配置摘要
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Fusion 训练配置摘要${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""

$PYTHON_ENV -c "
import config_fred
print(f'模态选择:        {config_fred.MODALITY}')
print(f'模型版本:        YOLOv5{config_fred.PHI}')
print(f'主干网络:        {config_fred.BACKBONE}')
print(f'输入尺寸:        {config_fred.INPUT_SHAPE}')
print(f'融合模式:        {config_fred.FUSION_MODE}')
print(f'压缩比率:        {config_fred.COMPRESSION_RATIO}')
print(f'冻结训练:        {config_fred.FREEZE_EPOCH} epochs')
print(f'解冻训练:        {config_fred.UNFREEZE_EPOCH} epochs')
print(f'冻结 batch size: {config_fred.FREEZE_BATCH_SIZE}')
print(f'解冻 batch size: {config_fred.UNFREEZE_BATCH_SIZE}')
print(f'优化器:          {config_fred.OPTIMIZER_TYPE.upper()}')
print(f'初始学习率:      {config_fred.INIT_LR}')
print(f'启用 Mosaic:     {config_fred.MOSAIC}')
print(f'启用 MixUp:      {config_fred.MIXUP}')
"

echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  日志保存位置${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  日志目录: $LOG_DIR${NC}"
echo -e "${GREEN}  TensorBoard: tensorboard --logdir $LOG_DIR${NC}"

# 训练选项
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  启动训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "${GREEN}开始训练...${NC}"
echo -e ""

# 运行训练脚本
cd "$PROJECT_ROOT"
$PYTHON_ENV train_fred_fusion.py "$@"

if [ $? -eq 0 ]; then
    echo -e ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  训练完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e ""
    echo -e "${GREEN}模型保存位置: $LOG_DIR${NC}"
    echo -e "${GREEN}TensorBoard 查看:${NC}"
    echo -e "${GREEN}  tensorboard --logdir $LOG_DIR${NC}"
    echo -e ""
else
    echo -e ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  训练失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}请检查错误信息${NC}"
    echo -e ""
    exit 1
fi