# FRED 数据集生成脚本 V2.0 - 快速开始

## 🚀 一键设置（推荐）

```bash
# 最简单的方式 - 一键完成所有设置
./setup_fred_dataset.sh
```

这将自动完成：
1. ✅ 检查 Python 环境
2. ✅ 检查 FRED 数据集
3. ✅ 转换为 COCO 格式
4. ✅ 验证生成的数据集
5. ✅ 显示下一步操作

---

## 📦 新版本特性

### 主要改进

| 特性 | 说明 |
|------|------|
| 🎯 **正确的标注文件** | 使用 `interpolated_coordinates.txt`（包含 drone_id） |
| 🔀 **两种划分模式** | 帧级别（推荐）+ 序列级别 |
| 🖼️ **智能图像源** | PADDED_RGB 优先，自动回退到 RGB |
| ✅ **完整验证** | 边界框、时间戳、坐标范围验证 |
| 📊 **详细统计** | 匹配率、序列分布等信息 |
| 📝 **完整文档** | 详细的使用指南和示例 |

---

## 📖 快速参考

### 手动转换

```bash
# 帧级别划分（推荐）
python convert_fred_to_coco_v2.py --split-mode frame --modality both

# 序列级别划分
python convert_fred_to_coco_v2.py --split-mode sequence --modality both

# 仅转换 RGB 模态
python convert_fred_to_coco_v2.py --modality rgb
```

### 验证数据集

```bash
python test_conversion_v2.py
```

### 开始训练

```bash
# RGB 模态
python train_fred.py --modality rgb

# Event 模态
python train_fred.py --modality event
```

---

## 📁 新增文件

```
yolov5-pytorch/
├── convert_fred_to_coco_v2.py              # 新版转换脚本
├── FRED_DATASET_CONVERSION_GUIDE.md        # 详细转换指南
├── test_conversion_v2.py                   # 测试脚本
├── setup_fred_dataset.sh                   # 一键设置脚本
├── DATASET_GENERATION_IMPROVEMENTS.md      # 改进总结
└── FRED_V2_README.md                       # 本文档
```

---

## 🔄 从旧版本迁移

```bash
# 1. 备份旧数据
mv datasets/fred_coco datasets/fred_coco_old

# 2. 使用新版本
./setup_fred_dataset.sh

# 3. 重新训练
python train_fred.py --modality rgb
```

---

## 📚 详细文档

- **转换指南**: `FRED_DATASET_CONVERSION_GUIDE.md`
- **改进总结**: `DATASET_GENERATION_IMPROVEMENTS.md`
- **项目文档**: `AGENTS.md`

---

## ❓ 常见问题

**Q: 为什么要使用新版本？**  
A: 新版本使用正确的标注文件（包含 drone_id），支持多目标追踪，数据质量更高。

**Q: 帧级别 vs 序列级别，选哪个？**  
A: 帧级别（推荐）- 数据分布更均衡；序列级别 - 严格场景分离。

**Q: 需要重新训练吗？**  
A: 建议重新训练，新版本数据质量更高。

**Q: 旧版本还能用吗？**  
A: 可以，但推荐使用新版本。

---

## ✅ 验证清单

转换完成后，确认：

- [ ] 所有 JSON 文件已生成（6 个）
- [ ] 图像数量合理（RGB: ~138k, Event: ~143k）
- [ ] 包含 drone_id 字段
- [ ] 测试脚本通过

---

**版本**: 2.0  
**更新**: 2025-11-01  
**维护**: YOLOv5-PyTorch 项目组
