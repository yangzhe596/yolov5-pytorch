# 工具脚本目录

本目录包含各种测试、验证和可视化工具脚本。

## 📋 脚本列表

### 测试脚本

#### `test_path_config.py`
测试 FRED 数据集路径配置是否正确

**用法**:
```bash
python scripts/test_path_config.py --modality rgb
python scripts/test_path_config.py --modality event
```

**功能**:
- 检查 FRED 根目录是否存在
- 检查 COCO 标注文件是否存在
- 验证所有图片路径是否正确
- 统计匹配率

---

#### `test_train_setup.py`
测试训练环境设置是否正确

**用法**:
```bash
python scripts/test_train_setup.py --modality rgb
```

**功能**:
- 检查 PyTorch 和 CUDA 环境
- 验证数据集完整性
- 测试数据加载器
- 检查模型权重

---

### 验证脚本

#### `verify_timestamp.py`
验证 FRED 数据集的时间戳对齐

**用法**:
```bash
python scripts/verify_timestamp.py --video_id 3
```

**功能**:
- 验证 RGB 和 Event 图像的时间戳
- 检查与 coordinates.txt 的对应关系
- 可视化时间戳分布

---

### 可视化脚本

#### `visualize_dataset.py`
可视化 FRED 数据集样本（主要工具）

**用法**:
```bash
# 可视化 RGB 训练集样本
python scripts/visualize_dataset.py --modality rgb --split train --num_samples 5

# 可视化 Event 测试集样本
python scripts/visualize_dataset.py --modality event --split test --num_samples 10

# 保存到指定目录
python scripts/visualize_dataset.py --modality rgb --split train --num_samples 5 --output_dir visualization/
```

**功能**:
- 加载 COCO 格式标注
- 绘制边界框
- 显示图像信息（尺寸、时间戳等）
- 保存可视化结果

---

#### `visualize_coco_samples.py`
COCO 样本可视化（简化版）

**用法**:
```bash
python scripts/visualize_coco_samples.py --modality rgb --split train
```

**功能**:
- 快速可视化 COCO 数据集
- 显示前几个样本

---

#### `visualize_multiple_samples.py`
批量可视化多个样本

**用法**:
```bash
python scripts/visualize_multiple_samples.py --modality rgb --num_samples 20
```

**功能**:
- 批量处理多个样本
- 生成网格布局的可视化结果

---

### 快捷脚本

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

#### `start_training.sh`
快速训练脚本

**用法**:
```bash
bash scripts/start_training.sh
```

**功能**:
- 交互式选择训练模态
- 自动配置训练参数
- 启动训练任务

---

## 🔧 使用建议

### 数据集准备阶段
1. 转换数据集后，使用 `test_path_config.py` 验证路径
2. 使用 `visualize_dataset.py` 检查数据质量
3. 使用 `verify_timestamp.py` 验证时间戳对齐

### 训练前检查
1. 使用 `test_train_setup.py` 检查训练环境
2. 使用 `visualize_dataset.py` 确认数据增强效果

### 训练后分析
1. 使用 `quick_eval.sh` 快速评估模型
2. 使用 `visualize_dataset.py` 查看预测结果

---

## 📝 注意事项

1. **Python 环境**: 所有脚本都需要使用正确的 Python 环境
   ```bash
   /home/yz/miniforge3/envs/torch/bin/python3 scripts/xxx.py
   ```

2. **工作目录**: 脚本应该从项目根目录运行
   ```bash
   cd /mnt/data/code/yolov5-pytorch
   python scripts/xxx.py
   ```

3. **路径配置**: 确保 FRED 数据集路径配置正确
   - 检查 `fred_config.py` 中的 `FRED_ROOT`
   - 或使用环境变量 `export FRED_ROOT=/path/to/fred`

---

**最后更新**: 2025-10-25
