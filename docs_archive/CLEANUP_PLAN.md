# 项目清理计划

## 📁 保留的核心文件

### 必需的脚本
- ✅ `train.py` - 标准 VOC 训练脚本
- ✅ `train_fred.py` - FRED 数据集训练脚本
- ✅ `predict.py` - 标准预测脚本
- ✅ `predict_fred.py` - FRED 数据集预测脚本
- ✅ `eval_fred.py` - FRED 数据集评估脚本
- ✅ `get_map.py` - mAP 评估脚本
- ✅ `yolo.py` - YOLO 类定义
- ✅ `voc_annotation.py` - VOC 数据集处理
- ✅ `convert_fred_to_coco.py` - FRED 数据集转换
- ✅ `kmeans_for_anchors.py` - 先验框计算
- ✅ `summary.py` - 模型结构查看

### 配置文件
- ✅ `config_fred.py` - FRED 训练配置
- ✅ `fred_config.py` - FRED 路径配置
- ✅ `requirements.txt` - 依赖项

### 核心文档
- ✅ `README.md` - 项目说明
- ✅ `AGENTS.md` - 开发指南
- ✅ `常见问题汇总.md` - 常见问题
- ✅ `FRED_PATH_CONFIG.md` - FRED 路径配置说明

---

## 🗑️ 可以清理的文件

### 测试/调试脚本（移至 scripts/ 目录）
- ❌ `test_path_config.py` - 路径测试脚本
- ❌ `test_train_setup.py` - 训练设置测试
- ❌ `verify_timestamp.py` - 时间戳验证
- ❌ `visualize_coco_samples.py` - COCO 样本可视化
- ❌ `visualize_dataset.py` - 数据集可视化
- ❌ `visualize_multiple_samples.py` - 批量可视化

### 快捷脚本（移至 scripts/ 目录）
- ❌ `quick_eval.sh` - 快速评估脚本
- ❌ `start_training.sh` - 快速训练脚本

### 临时文档（移至 docs_archive/ 目录）
- ❌ `FINAL_STATUS.txt` - 临时状态文件
- ❌ `SUMMARY.txt` - 临时总结文件

---

## 📂 建议的目录结构

```
yolov5-pytorch/
├── README.md                    # 项目说明
├── AGENTS.md                    # 开发指南
├── 常见问题汇总.md              # 常见问题
├── FRED_PATH_CONFIG.md          # FRED 路径配置
├── requirements.txt             # 依赖项
│
├── train.py                     # 标准训练
├── train_fred.py                # FRED 训练
├── predict.py                   # 标准预测
├── predict_fred.py              # FRED 预测
├── eval_fred.py                 # FRED 评估
├── get_map.py                   # mAP 评估
├── yolo.py                      # YOLO 类
├── voc_annotation.py            # VOC 处理
├── convert_fred_to_coco.py      # FRED 转换
├── kmeans_for_anchors.py        # 先验框计算
├── summary.py                   # 模型结构
│
├── config_fred.py               # FRED 训练配置
├── fred_config.py               # FRED 路径配置
│
├── scripts/                     # 工具脚本
│   ├── test_path_config.py
│   ├── test_train_setup.py
│   ├── verify_timestamp.py
│   ├── visualize_dataset.py
│   ├── visualize_coco_samples.py
│   ├── visualize_multiple_samples.py
│   ├── quick_eval.sh
│   └── start_training.sh
│
├── docs_archive/                # 归档文档
│   ├── FINAL_STATUS.txt
│   ├── SUMMARY.txt
│   └── (其他已归档的文档)
│
├── nets/                        # 网络模型
├── utils/                       # 工具函数
├── model_data/                  # 模型权重
├── datasets/                    # 数据集
└── logs/                        # 训练日志
```
