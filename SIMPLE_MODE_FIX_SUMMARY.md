# simple 模式 bug 修复总结

## 问题描述

在启用 simple 模式运行时，出现以下错误：

```
❌ 错误: get_frame_split() got an unexpected keyword argument 'seed'
Traceback (most recent call last):
  File "convert_fred_to_fusion_v2.py", line 955, in main
    converter.generate_all_fusion(
  File "convert_fred_to_fusion_v2.py", line 796, in generate_all_fusion
    split_results = self.generate_frame_level_split(sequences, train_ratio, val_ratio)
  File "convert_fred_to_fusion_v2.py", line 619, in generate_frame_level_split
    validation_result = self.validator.validate_frame_split(sequences)
  File "convert_fred_to_fusion_v2.py", line 152, in validate_frame_split
    split, rand_val = self.get_frame_split(seq_id, frame_info, seed=seed)
TypeError: get_frame_split() got an unexpected keyword argument 'seed
```

## 根本原因

### 1. 方法签名不匹配

`FrameSplitValidator.get_frame_split()` 方法的定义：

```python
def get_frame_split(self, seq_id, frame_info, train_ratio=0.7, val_ratio=0.15):
    """
    确定帧的数据集划分
    
    Args:
        seq_id: 序列 ID
        frame_info: 帧信息字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        str: 'train', 'val', 或 'test'
    """
```

**问题**：方法没有 `seed` 参数，但是从另一个文件中复制时保留了这个参数。

### 2. 参数不一致的来源

在 `convert_fred_to_fusion.py` 中，`get_frame_split` 的定义是：

```python
def get_frame_split(self, frame_key, seed=42):
    """
    根据帧标识符和种子确定数据集划分
    
    Args:
        frame_key: 帧标识符
        seed: 随机种子
        
    Returns:
        str: 'train', 'val', 或 'test'
    """
```

但在 `convert_fred_to_fusion_v2.py` 中，设计是使用 `self.seed` 类属性而不是参数传递。

### 3. 调用方式

```python
# 调用处（有 bug）
split, rand_val = self.get_frame_split(seq_id, frame_info, seed=seed)

# 实现处（没有 seed 参数）
def get_frame_split(self, seq_id, frame_info, train_ratio=0.7, val_ratio=0.15):
```

## 修复方案

### 方案选择

有两个可选方案：

1. **修改方法签名**：添加 `seed` 参数
2. **移除调用参数**：使用 `self.seed`（与现有逻辑一致）

**选择了方案 2**，因为：
- 保持与 `get_frame_split` 方法内部的一致性（使用 `self.seed`）
- 减少代码修改范围
- 方法内部已经使用 `hash_input = f"{frame_key}_{self.seed}"`

### 修复步骤

#### 1. 移除参数传递<tool_call>
<function=read_file>
<parameter=absolute_path>/mnt/data/code/yolov5-pytorch/convert_fred_to_fusion_v2.py