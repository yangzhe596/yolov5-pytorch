#!/bin/bash
# 可视化特定图片的快捷脚本

PYTHON=/home/yz/miniforge3/envs/torch/bin/python3

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT=$SCRIPT_DIR/visualize_specific_images.py

echo "========================================"
echo "可视化特定图片工具"
echo "========================================"
echo ""

# 显示菜单
echo "请选择操作:"
echo "1) 列出可用图片（RGB训练集，前20张）"
echo "2) 列出可用图片（Event训练集，前20张）"
echo "3) 通过图片ID可视化"
echo "4) 通过文件名可视化"
echo "5) 通过序列号可视化"
echo "6) 通过正则表达式可视化"
echo "0) 自定义命令"
echo ""
read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo "列出 RGB 训练集前20张图片..."
        $PYTHON $SCRIPT --modality rgb --split train --list --list_limit 20
        ;;
    2)
        echo "列出 Event 训练集前20张图片..."
        $PYTHON $SCRIPT --modality event --split train --list --list_limit 20
        ;;
    3)
        echo ""
        read -p "模态 (rgb/event) [rgb]: " modality
        modality=${modality:-rgb}
        read -p "划分 (train/val/test) [train]: " split
        split=${split:-train}
        read -p "图片ID（空格分隔，如: 1 2 3）: " ids
        
        if [ -z "$ids" ]; then
            echo "❌ 请输入至少一个图片ID"
            exit 1
        fi
        
        echo "可视化图片 ID: $ids"
        $PYTHON $SCRIPT --modality $modality --split $split --image_ids $ids
        ;;
    4)
        echo ""
        read -p "模态 (rgb/event) [rgb]: " modality
        modality=${modality:-rgb}
        read -p "划分 (train/val/test) [train]: " split
        split=${split:-train}
        read -p "文件名关键词（空格分隔，如: Video_0_16_03_03 Video_1）: " filenames
        
        if [ -z "$filenames" ]; then
            echo "❌ 请输入至少一个文件名关键词"
            exit 1
        fi
        
        echo "可视化包含关键词的图片: $filenames"
        $PYTHON $SCRIPT --modality $modality --split $split --filenames $filenames
        ;;
    5)
        echo ""
        read -p "模态 (rgb/event) [rgb]: " modality
        modality=${modality:-rgb}
        read -p "划分 (train/val/test) [train]: " split
        split=${split:-train}
        read -p "序列号（空格分隔，如: 1 3 5）: " sequences
        
        if [ -z "$sequences" ]; then
            echo "❌ 请输入至少一个序列号"
            exit 1
        fi
        
        echo "可视化序列: $sequences"
        $PYTHON $SCRIPT --modality $modality --split $split --sequences $sequences
        ;;
    6)
        echo ""
        read -p "模态 (rgb/event) [rgb]: " modality
        modality=${modality:-rgb}
        read -p "划分 (train/val/test) [train]: " split
        split=${split:-train}
        read -p "正则表达式（如: Video_0_16_03_.*）: " pattern
        
        if [ -z "$pattern" ]; then
            echo "❌ 请输入正则表达式"
            exit 1
        fi
        
        echo "可视化匹配正则表达式的图片: $pattern"
        $PYTHON $SCRIPT --modality $modality --split $split --pattern "$pattern"
        ;;
    0)
        echo ""
        echo "请输入完整命令（不包括 python 和脚本名）"
        echo "示例: --modality rgb --split train --image_ids 1 2 3"
        read -p "命令: " custom_cmd
        
        if [ -z "$custom_cmd" ]; then
            echo "❌ 命令不能为空"
            exit 1
        fi
        
        echo "执行自定义命令..."
        $PYTHON $SCRIPT $custom_cmd
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo ""
echo "查看结果:"
echo "  输出目录: specific_visualization/"
echo ""
