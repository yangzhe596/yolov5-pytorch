#!/bin/bash
# FRED-YOLOv5 训练启动脚本

echo "=========================================="
echo "FRED-YOLOv5 训练启动"
echo "=========================================="
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: bash start_training.sh [rgb|event] [选项]"
    echo ""
    echo "示例:"
    echo "  bash start_training.sh rgb              # RGB模态，带mAP评估"
    echo "  bash start_training.sh event            # Event模态，带mAP评估"
    echo "  bash start_training.sh rgb --no_eval    # RGB模态，不评估mAP（快速）"
    echo "  bash start_training.sh event --no_eval  # Event模态，不评估mAP（快速）"
    echo ""
    exit 1
fi

MODALITY=$1
NO_EVAL=""

if [ "$2" == "--no_eval" ]; then
    NO_EVAL="--no_eval_map"
fi

# 验证模态
if [ "$MODALITY" != "rgb" ] && [ "$MODALITY" != "event" ]; then
    echo "❌ 错误: 模态必须是 'rgb' 或 'event'"
    exit 1
fi

# 检查数据集
DATASET_PATH="datasets/fred_coco/$MODALITY"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET_PATH"
    echo ""
    echo "请先运行数据集转换:"
    echo "  python convert_fred_to_coco.py --modality $MODALITY"
    exit 1
fi

# 检查训练集
TRAIN_JSON="$DATASET_PATH/annotations/instances_train.json"
if [ ! -f "$TRAIN_JSON" ]; then
    echo "❌ 错误: 训练集标注文件不存在: $TRAIN_JSON"
    exit 1
fi

# 显示配置
echo "📊 训练配置:"
echo "  模态: ${MODALITY^^}"
echo "  数据集: $DATASET_PATH"
if [ -z "$NO_EVAL" ]; then
    echo "  mAP评估: ✅ 启用（每10个epoch）"
    echo "  评估数据集: 测试集"
else
    echo "  mAP评估: ❌ 禁用（快速训练）"
fi
echo ""

# 统计数据集
TRAIN_COUNT=$(python -c "import json; data=json.load(open('$TRAIN_JSON')); print(len(data['images']))")
VAL_JSON="$DATASET_PATH/annotations/instances_val.json"
VAL_COUNT=$(python -c "import json; data=json.load(open('$VAL_JSON')); print(len(data['images']))")
TEST_JSON="$DATASET_PATH/annotations/instances_test.json"
TEST_COUNT=$(python -c "import json; data=json.load(open('$TEST_JSON')); print(len(data['images']))")

echo "📁 数据集统计:"
echo "  训练集: $TRAIN_COUNT 张图片"
echo "  验证集: $VAL_COUNT 张图片"
echo "  测试集: $TEST_COUNT 张图片"
echo ""

# 创建日志目录
LOG_DIR="logs/fred_$MODALITY"
mkdir -p "$LOG_DIR"

# 询问是否继续
read -p "是否开始训练? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "❌ 训练已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "🚀 开始训练..."
echo "=========================================="
echo ""

# 训练命令
TRAIN_CMD="python train_fred.py --modality $MODALITY $NO_EVAL"

echo "执行命令: $TRAIN_CMD"
echo ""

# 执行训练
$TRAIN_CMD

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练完成！"
    echo "=========================================="
    echo ""
    echo "📁 训练输出:"
    echo "  日志目录: $LOG_DIR"
    echo "  最佳模型: $LOG_DIR/fred_${MODALITY}_best.pth"
    echo "  最终模型: $LOG_DIR/fred_${MODALITY}_final.pth"
    echo ""
    echo "📊 查看训练曲线:"
    echo "  tensorboard --logdir $LOG_DIR"
    echo ""
    echo "🔍 评估模型:"
    echo "  python eval_fred.py --modality $MODALITY"
    echo ""
    echo "🎯 测试预测:"
    echo "  python predict_fred.py --modality $MODALITY"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败！"
    echo "=========================================="
    echo ""
    echo "请检查错误信息并重试"
    exit 1
fi
