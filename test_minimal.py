#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化测试 - 检查为什么 main 函数没有被调用
"""
print("脚本开始", flush=True)

def main():
    print("main 函数执行", flush=True)

if __name__ == "__main__":
    print("进入主程序", flush=True)
    main()
    print("脚本结束", flush=True)

# 在模块级别添加一些代码
print("模块级别代码执行", flush=True)