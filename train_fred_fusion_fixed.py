#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版本的 train_fred_fusion.py
"""
import sys
import os

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始脚本
import train_fred_fusion

# 直接调用 main 函数
if __name__ == "__main__":
    print("开始执行 Fusion 训练脚本...", flush=True)
    train_fred_fusion.main()
    print("执行完成！", flush=True)