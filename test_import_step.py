#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐步导入 train_fred_fusion.py 的模块，找出问题
"""
import sys
import os

print("开始逐步导入...", flush=True)

# 1. 导入标准库
print("\n1. 导入标准库...", flush=True)
import argparse
import datetime
import os
import sys
import time
from copy import deepcopy as copy
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
print("✓ 标准库导入完成", flush=True)

# 2. 导入第三方库
print("\n2. 导入第三方库...", flush=True)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
print("✓ 第三方库导入完成", flush=True)

# 3. 导入项目模块
print("\n3. 导入项目模块...", flush=True)
from nets.yolo_fusion import YoloFusionBody
from nets.yolo_training import ModelEMA, YOLOLoss, get_lr_scheduler
from utils.callbacks import LossHistory
from utils.callbacks_coco import CocoEvalCallback
from utils.dataloader_fusion import FusionYoloDataset, fusion_dataset_collate
from utils.utils import (get_anchors, get_classes, get_lr, seed_everything,
                         show_config, worker_init_fn)
import config_fred
print("✓ 项目模块导入完成", flush=True)

# 4. 设置全局变量
print("\n4. 设置全局变量...", flush=True)
fred_root = "/mnt/data/datasets/fred"
globals().update(vars(config_fred))
print("✓ 全局变量设置完成", flush=True)

# 5. 定义函数
print("\n5. 定义函数...", flush=True)
exec(open('train_fred_fusion.py').read().split('def main')[0])
print("✓ 函数定义完成", flush=True)

print("\n所有导入和定义完成，准备执行 main 函数", flush=True)