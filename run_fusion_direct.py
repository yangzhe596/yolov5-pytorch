#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接执行 train_fred_fusion.py 的 main 函数
"""
import sys
import os

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置命令行参数
sys.argv = ['train_fred_fusion.py', '--compression_ratio', '1.0', '--modality', 'dual']

# 导入并执行
print("导入 train_fred_fusion 模块...", flush=True)
import train_fred_fusion

print("执行 main 函数...", flush=True)
train_fred_fusion.main()

print("执行完成！", flush=True)