#!/bin/bash
# 数据集可视化验证快速启动脚本

echo "========================================"
echo "数据集可视化验证工具"
echo "========================================"
echo ""

# 设置Python环境
PYTHON=/home/yz/miniforge3/envs/torch/bin/python3

# 默认参数
COCO_ROOT="/mnt/data/datasets/fred"
OUTPUT_ROOT="datasets/fred_coco"
OUTPUT_DIR="dataset_validation"
NUM_SAMPLES=20

# 显示菜单
echo "请选择验证模式:"
echo "1) RGB训练集 (20个样本)"
echo "2) RGB验证集 (20个样本)"
echo "3) RGB测试集 (20个样本)"
echo "4) Event训练集 (20个样本)"
echo "5) Event验证集 (20个样本)"
echo "6) Event测试集 (20个样本)"
echo "7) RGB所有划分 (train/val/test)"
echo "8) Event所有划分 (train/val/test)"
echo "9) 全部验证 (RGB+Event, train/val/test)"
echo "0) 自定义参数"
echo ""
read -p "请输入选项 [1-9, 0]: " choice

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

case $choice in
    1)
        echo "验证 RGB 训练集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality rgb --split train --num_samples $NUM_SAMPLES
        ;;
    2)
        echo "验证 RGB 验证集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality rgb --split val --num_samples $NUM_SAMPLES
        ;;
    3)
        echo "验证 RGB 测试集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality rgb --split test --num_samples $NUM_SAMPLES
        ;;
    4)
        echo "验证 Event 训练集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality event --split train --num_samples $NUM_SAMPLES
        ;;
    5)
        echo "验证 Event 验证集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality event --split val --num_samples $NUM_SAMPLES
        ;;
    6)
        echo "验证 Event 测试集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality event --split test --num_samples $NUM_SAMPLES
        ;;
    7)
        echo "验证 RGB 所有划分..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality rgb --split all --num_samples $NUM_SAMPLES
        ;;
    8)
        echo "验证 Event 所有划分..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality event --split all --num_samples $NUM_SAMPLES
        ;;
    9)
        echo "验证全部数据集..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality both --split all --num_samples $NUM_SAMPLES
        ;;
    0)
        echo ""
        read -p "模态 (rgb/event/both): " modality
        read -p "划分 (train/val/test/all): " split
        read -p "样本数量: " num_samples
        echo "执行自定义验证..."
        $PYTHON $SCRIPT_DIR/visualize_dataset_validation.py --modality $modality --split $split --num_samples $num_samples
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "验证完成！"
echo "========================================"
echo ""
echo "输出目录: $OUTPUT_DIR/"
echo ""
echo "查看结果:"
echo "  1. 可视化图片: $OUTPUT_DIR/<modality>_<split>/*.jpg"
echo "  2. JSON报告: $OUTPUT_DIR/<modality>_<split>/validation_report.json"
echo "  3. HTML报告: $OUTPUT_DIR/<modality>_<split>/validation_report.html"
echo ""
echo "在浏览器中打开HTML报告以查看完整的可视化结果！"
echo ""
