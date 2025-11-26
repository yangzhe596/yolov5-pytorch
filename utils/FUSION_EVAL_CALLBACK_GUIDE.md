# FusionCocoEvalCallback 文档

## 概述

`FusionCocoEvalCallback` 是专门为 **Fusion 模型**（双模态目标检测模型）设计的 COCO 格式评估回调，继承自 `CocoEvalCallback` 并添加了对 **RGB + Event 双模态输入** 的支持。

## 主要特性

- ✅ **继承自完整的 `CocoEvalCallback`**：保留所有 COCO 评估功能
- ✅ **支持多种评估模式**：
  - `rgb_only`：只使用 RGB 模态评估（默认，速度最快）
  - `event_only`：只使用 Event 模态评估
  - `dual_avg`：使用双模态平均值评估
  - `dual_concat`：双模态拼接（需模型支持）
- ✅ **自动处理双模态图片路径**：从 COCO JSON 中自动提取图片信息
- ✅ **快速验证模式**：支持限制评估样本数量（`max_eval_samples`）
- ✅ **显存优化**：使用混合精度和显存清理
- ✅ **详细的评估日志**：生成 mAP 曲线、记录最佳模型

## 快速开始

### 1. 导入模块

```python
from utils.callbacks_fusion import FusionCocoEvalCallback
```

### 2. 创建评估回调

```python
# Fusion 模型配置
input_shape = [640, 640]
class_names = ['object']  # FRED 数据集只有一个类别
num_classes = len(class_names)

# COCO 数据集路径
val_json = 'datasets/fred_coco/rgb/annotations/instances_val.json'
image_dir_rgb = 'datasets/fred_coco/rgb/val'
image_dir_event = 'datasets/fred_coco/event/val'

# 创建回调
eval_callback = FusionCocoEvalCallback(
    net=model,
    input_shape=input_shape,
    anchors=anchors,
    anchors_mask=anchors_mask,
    class_names=class_names,
    num_classes=num_classes,
    coco_json_path=val_json,
    image_dir_rgb=image_dir_rgb,
    image_dir_event=image_dir_event,
    log_dir='logs/fred_fusion',
    cuda=True,
    confidence=0.05,
    nms_iou=0.5,
    letterbox_image=True,
    MINOVERLAP=0.5,
    eval_flag=True,
    period=1,  # 每个 epoch 评估一次
    max_eval_samples=1000,  # 快速验证模式（可选）
    fusion_mode="rgb_only"  # Fusion 模式
)
```

### 3. 在训练循环中使用

```python
for epoch in range(num_epochs):
    # 训练模型...
    train_one_epoch(model, ...)
    
    # 评估模型
    eval_callback.on_epoch_end(epoch, model)
```

## 参数说明

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `net` | nn.Module | Fusion 模型 |
| `input_shape` | list | 输入尺寸 [H, W] |
| `anchors` | np.array | 先验框 |
| `anchors_mask` | list | 先验框 mask |
| `class_names` | list | 类别名称列表 |
| `num_classes` | int | 类别数量 |
| `coco_json_path` | str | COCO 标注 JSON 路径 |
| `image_dir_rgb` | str | RGB 图片目录 |
| `image_dir_event` | str | Event 图片目录 |
| `log_dir` | str | 日志目录 |
| `cuda` | bool | 是否使用 CUDA |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `map_out_path` | `.temp_map_out` | mAP 计算临时目录 |
| `max_boxes` | 100 | 最大检测框数量 |
| `confidence` | 0.05 | 置信度阈值 |
| `nms_iou` | 0.5 | NMS IOU 阈值 |
| `letterbox_image` | True | 是否使用 letterbox |
| `MINOVERLAP` | 0.5 | mAP@0.5 |
| `eval_flag` | True | 是否进行评估 |
| `period` | 1 | 评估周期（每 N 个 epoch） |
| `max_eval_samples` | None | 最大评估样本数（用于快速验证） |
| `fusion_mode` | `"rgb_only"` | Fusion 评估模式 |

## Fusion 评估模式

### 1. RGB Only (推荐)

```python
fusion_mode="rgb_only"
```

- ✅ 只使用 RGB 模态进行评估
- ✅ 速度快（只做一次前向传播）
- ✅ 适合训练初期和快速验证
- ⚠️ 未充分利用 Event 模态信息

### 2. Event Only

```python
fusion_mode="event_only"
```

- ✅ 只使用 Event 模态进行评估
- ✅ 适合分析 Event 模态的性能
- ⚠️ 需要完整的 Event 图片数据集
- ⚠Event 图片可能丢失 RGB 的纹理信息

### 3. Dual Average

```python
fusion_mode="dual_avg"
```

- ✅ 同时使用 RGB 和 Event 模态
- ✅ 调用模型两次（RGB + Event）
- ⚠️ 评估时间 x2
- ⚠️ 需要验证模式是否满足需求

## 输出文件

评估完成后，会在 `log_dir` 中生成以下文件：

```
logs/fred_fusion/
├── epoch_map.txt      # mAP 记录
├── epoch_map.png      # mAP 曲线
└── events.out.*       # TensorBoard 日志（如果有）
```

## 最佳实践

### 1. 训练初期（快速迭代）

```python
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    fusion_mode="rgb_only",      # 使用 RGB only 模式
    max_eval_samples=1000,       # 限制评估样本
    period=5,                    # 每 5 个 epoch 评估一次
)
```

### 2. 训练中期（优化性能）

```python
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    fusion_mode="rgb_only",      # 继续使用 RGB only
    max_eval_samples=None,       # 使用全部样本
    period=2,                    # 每 2 个 epoch 评估一次
)
```

### 3. 训练末期（精细调优）

```python
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    fusion_mode="rgb_only",      # 或 "dual_avg"（时间允许）
    max_eval_samples=None,       # 使用全部样本
    period=1,                    # 每个 epoch 都评估
)
```

## 常见问题

### Q1: 如何切换评估模式？

在创建 `FusionCocoEvalCallback` 时设置 `fusion_mode` 参数：

```python
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    fusion_mode="rgb_only"  # 可选: "rgb_only", "event_only", "dual_avg"
)
```

### Q2: 评估速度太慢怎么办？

1. 使用 `rgb_only` 模式（最快）
2. 设置 `max_eval_samples` 限制评估样本数量
3. 增加 `period` 参数，减少评估频率

```python
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    fusion_mode="rgb_only",
    max_eval_samples=500,   # 限制样本
    period=2,               # 增加周期
)
```

### Q3: 如何保存最佳模型？

`FusionCocoEvalCallback` 会自动记录 mAP 并保存最佳模型权重：

```python
# 在训练循环中
if eval_callback.maps[-1] > eval_callback.best_map:
    eval_callback.best_map = eval_callback.maps[-1]
    torch.save(model.state_dict(), 'best_model.pth')
```

### Q4: Event 图片找不到怎么办？

确保 `image_dir_event` 路径正确，并且图片格式匹配（通常是 `.png`）：

```bash
# 检查 Event 图片
ls datasets/fred_coco/event/val/ | head -10

# 示例输出
# 000001.png
# 000002.png
# ...
```

### Q5: 图片 ID 如何匹配？

`FusionCocoEvalCallback` 使用图片文件名（不含扩展名）作为 ID：

- RGB 图片路径：`datasets/fred_coco/rgb/val/000001.jpg`
- Event 图片路径：`datasets/fred_coco/event/val/000001.png`
- 图片 ID：`000001`

## 与原始 `CocoEvalCallback` 的区别

| 特性 | CocoEvalCallback | FusionCocoEvalCallback |
|------|------------------|------------------------|
| 输入模态 | 单模态（RGB） | 双模态（RGB + Event） |
| 调用方式 | `model(image)` | `model(rgb, event)` |
| 图片加载 | 仅 image_dir | image_dir_rgb + image_dir_event |
| 评估模式 | N/A | 支持多种 Fusion 模式 |
| 显存优化 | 标准 | 混合精度 + 显存清理 |
| 适用场景 | 标准 YOLOv5 | Fusion 双模态模型 |

## 参考示例

详细示例请参考 `utils/callbacks_fusion_example.py`：

```python
python -m utils.callbacks_fusion_example
```

## 相关文档

- [FUSION兼容性修复说明](FUSION_COMPATIBILITY_FIX.md)
- [Fusion数据加载器指南](FUSION_DATALOADER_GUIDE.md)
- [FRED数据集快速开始](QUICK_START_FRED.md)

## 更新日志

### v1.0 (2025-11-26)

- ✅ 首次发布 FusionCocoEvalCallback
- ✅ 支持 rgb_only、event_only、dual_avg 三种 Fusion 评估模式
- ✅ 继承完整完整的 CocoEvalCallback 功能
- ✅ 优化显存使用和评估速度
- ✅ 添加详细的使用示例和文档