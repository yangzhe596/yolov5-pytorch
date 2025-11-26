#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细测试 - 逐步导入并检查每个步骤
"""
import sys
import os

print("="*60)
print("开始详细测试")
print("="*60)

# 1. 导入 config_fred
print("\n1. 导入 config_fred...", flush=True)
import config_fred
print("✓ config_fred 导入成功", flush=True)

# 2. 检查 config_fred 的内容
print("\n2. 检查 config_fred 的内容...", flush=True)
print(f"NUM_CLASSES: {config_fred.NUM_CLASSES}")
print(f"CLASS_NAMES: {config_fred.CLASS_NAMES}")
print("✓ config_fred 内容检查完成", flush=True)

# 3. 设置全局变量
print("\n3. 设置全局变量...", flush=True)
fred_root = "/mnt/data/datasets/fred"
globals().update(vars(config_fred))
print("✓ 全局变量设置完成", flush=True)

# 4. 检查是否有模块级别的代码
print("\n4. 检查 train_fred_fusion.py 的模块级别代码...", flush=True)
with open('train_fred_fusion.py', 'r') as f:
    lines = f.readlines()

# 查找不在函数或类定义中的代码
in_function = False
in_class = False
indent_level = 0
module_level_code = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue
    
    # 检查是否是函数或类定义
    if stripped.startswith('def ') or stripped.startswith('class '):
        continue
    
    # 检查缩进
    current_indent = len(line) - len(line.lstrip())
    
    # 如果是模块级别的代码（缩进为0或很小）
    if current_indent <= 4 and not in_function and not in_class:
        if stripped and not stripped.startswith('import') and not stripped.startswith('from'):
            module_level_code.append((i+1, stripped))

if module_level_code:
    print("发现模块级别的代码:")
    for line_num, code in module_level_code[:10]:  # 只显示前10个
        print(f"  行 {line_num}: {code}")
else:
    print("没有发现模块级别的执行代码")

print("✓ 模块级别代码检查完成", flush=True)

print("\n" + "="*60)
print("详细测试完成")
print("="*60)