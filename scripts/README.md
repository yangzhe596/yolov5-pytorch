# 工具脚本目录

本目录包含各种训练、评估和可视化的辅助脚本。

## 📋 脚本列表

### 训练脚本

#### `start_training.sh`
快速训练启动脚本

**用法**:
```bash
bash scripts/start_training.sh
```

**功能**:
- 交互式选择训练模态（RGB/Event）
- 自动配置训练参数
- 启动训练任务

---

### 评估脚本

#### `quick_eval.sh`
快速评估脚本

**用法**:
```bash
bash scripts/quick_eval.sh
```

**功能**:
- 快速评估 RGB 和 Event 模型
- 自动选择最佳权重
- 生成评估报告

---

### 可视化脚本

#### `validate_dataset.sh`
数据集可视化验证快速启动脚本

**用法**:
```bash
bash scripts/validate_dataset.sh
```

**功能**:
- 交互式选择验证模式
- 支持 RGB/Event 模态
- 支持 train/val/test 划分
- 调用 `visualize_dataset_validation.py` 进行验证

**验证模式**:
1. RGB训练集 (20个样本)
2. RGB验证集 (20个样本)
3. RGB测试集 (20个样本)
4. Event训练集 (20个样本)
5. Event验证集 (20个样本)
6. Event测试集 (20个样本)
7. RGB所有划分 (train/val/test)
8. Event所有划分 (train/val/test)
9. 全部验证 (RGB+Event, train/val/test)
0. 自定义参数

**输出**:
- 可视化图片: `dataset_validation/<modality>_<split>/*.jpg`
- JSON报告: `dataset_validation/<modality>_<split>/validation_report.json`
- HTML报告: `dataset_validation/<modality>_<split>/validation_report.html` ⭐

---

#### `visualize_images.sh`
可视化特定图片的快捷脚本

**用法**:
```bash
bash scripts/visualize_images.sh
```

**功能**:
- 交互式选择可视化方式
- 支持多种筛选方式
- 调用 `visualize_specific_images.py` 进行可视化

**可视化方式**:
1. 列出可用图片（RGB训练集，前20张）
2. 列出可用图片（Event训练集，前20张）
3. 通过图片ID可视化
4. 通过文件名可视化
5. 通过序列号可视化
6. 通过正则表达式可视化
0. 自定义命令

**输出**:
- 可视化图片: `specific_visualization/<modality>_<split>/*.jpg`

---

## 🔧 使用建议

### 数据集准备阶段
1. 转换数据集后，使用 `validate_dataset.sh` 验证路径和标注质量
2. 使用 `visualize_images.sh` 检查特定图片的标注

### 训练阶段
1. 使用 `start_training.sh` 快速启动训练
2. 训练过程中可以使用 TensorBoard 监控

### 训练后分析
1. 使用 `quick_eval.sh` 快速评估模型
2. 使用可视化脚本查看预测结果

---

## 📝 注意事项

1. **Python 环境**: 所有脚本都需要使用正确的 Python 环境
   ```bash
   /home/yz/miniforge3/envs/torch/bin/python3
   ```

2. **工作目录**: 脚本应该从项目根目录运行
   ```bash
   cd /mnt/data/code/yolov5-pytorch
   bash scripts/xxx.sh
   ```

3. **路径配置**: 确保 FRED 数据集路径配置正确
   - 检查脚本中的 `COCO_ROOT` 和 `OUTPUT_ROOT` 变量
   - 或使用环境变量 `export FRED_ROOT=/path/to/fred`

---

## 🎯 核心可视化工具

项目根目录下有三个核心可视化 Python 脚本：

1. **visualize_dataset_validation.py** - 数据集质量验证
   - 随机抽样验证
   - 边界框质量检查
   - 生成 HTML 报告
   - **训练前必用**

2. **visualize_specific_images.py** - 特定图片可视化
   - 按 ID/文件名/序列号筛选
   - 正则表达式匹配
   - 列出可用图片

3. **visualize_fred_sequences.py** - 序列视频生成
   - 导出 MP4 视频
   - RGB vs Event 对比
   - 高性能处理（90+ FPS）

---

**最后更新**: 2025-11-01
