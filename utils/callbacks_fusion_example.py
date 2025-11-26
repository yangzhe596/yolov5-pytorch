"""
FusionCocoEvalCallback 使用示例

该文件演示如何在 Fusion 模型训练中集成评估回调
"""
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

# 添加 utils 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.callbacks_fusion import FusionCocoEvalCallback
from utils.utils import get_anchors
from models.fusion_yolo import FusionYOLO  # 假设 Fusion 模型定义在此


def create_fusion_model():
    """创建 Fusion 模型示例"""
    # Fusion 模型参数
    input_shape = [640, 640]
    backbone = 'cspdarknet'
    phi = 's'
    num_classes = 1
    anchors_path = 'datasets/nps/anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # 获取 anchors
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 创建模型（这里使用伪代码，实际需要你的 Fusion 模型）
    model = FusionYOLO(
        num_classes=num_classes,
        backbone=backbone,
        phi=phi,
        input_shape=input_shape
    )
    
    return model, input_shape, anchors, anchors_mask


def setup_fusion_eval_callback(model, device, fusion_mode="rgb_only"):
    """设置 Fusion 评估回调"""
    
    # 数据集配置
    class_names = ['object']  # FRED 数据集只有一个类别
    
    # COCO JSON 文件路径
    val_json = 'datasets/fred_coco/rgb/annotations/instances_val.json'
    
    # 图片目录
    image_dir_rgb = 'datasets/fred_coco/rgb/val'
    image_dir_event = 'datasets/fred_coco/event/val'
    
    # 日志目录
    log_dir = 'logs/fred_fusion_test'
    os.makedirs(log_dir, exist_ok=True)
    
    # 模型输入配置
    input_shape = [640, 640]
    anchors_path = 'datasets/nps/anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors, num_anchors = get_anchors(anchors_path)
    
    # 创建评估回调
    eval_callback = FusionCocoEvalCallback(
        net=model,
        input_shape=input_shape,
        anchors=anchors,
        anchors_mask=anchors_mask,
        class_names=class_names,
        num_classes=len(class_names),
        coco_json_path=val_json,
        image_dir_rgb=image_dir_rgb,
        image_dir_event=image_dir_event,
        log_dir=log_dir,
        cuda=(device.type == 'cuda'),
        confidence=0.05,
        nms_iou=0.5,
        letterbox_image=True,
        MINOVERLAP=0.5,
        eval_flag=True,
        period=1,  # 每个 epoch 都评估
        max_eval_samples=1000,  # 快速验证模式
        fusion_mode=fusion_mode  # Fusion 模式
    )
    
    return eval_callback


def main():
    """主函数：演示完整的训练流程"""
    
    # 1. 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 创建 Fusion 模型
    print("创建 Fusion 模型...")
    model, input_shape, anchors, anchors_mask = create_fusion_model()
    model = model.to(device)
    
    # 3. 设置优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.MSELoss()  # 用于示例，实际使用 YOLO 损失
    
    # 4. 创建评估回调（使用 RGB_only 模式，默认最快）
    print("创建评估回调...")
    eval_callback = setup_fusion_eval_callback(
        model=model,
        device=device,
        fusion_mode="rgb_only"  # 可选: "rgb_only" | "event_only" | "dual_avg"
    )
    
    # 5. 模拟训练循环
    num_epochs = 10
    print(f"\n开始训练 FUSION 模型: {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        model.train()
        
        # 模拟训练过程
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练代码...
        # for batch_idx, (images, targets) in enumerate(train_loader):
        #     # 前向传播
        #     outputs = model(rgb_images, event_images)
        #     loss = criterion(outputs, targets)
        #     
        #     # 反向传播
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        # 评估阶段
        if epoch % 1 == 0:  # 每个 epoch 评估一次
            model.eval()
            print(f"\n[评估] Epoch {epoch+1}")
            eval_callback.on_epoch_end(epoch, model)


def test_fusion_modes():
    """测试不同的 Fusion 评估模式"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, anchors, anchors_mask = create_fusion_model()
    model = model.to(device)
    
    class_names = ['object']
    input_shape = [640, 640]
    
    vanilla_eval = {
        "name": "RGB Only",
        "mode": "rgb_only",
        "description": "只使用 RGB 模态评估，速度最快"
    }
    
    event_eval = {
        "name": "Event Only",
        "mode": "event_only",
        "description": "只使用 Event 模态评估"
    }
    
    dual_eval = {
        "name": "Dual Average",
        "mode": "dual_avg",
        "description": "使用双模态信息评估"
    }
    
    print("Fusion 评估模式对比:")
    print("=" * 60)
    
    for config in [vanilla_eval, event_eval, dual_eval]:
        print(f"\n{config['name']}:")
        print(f"  模式: {config['mode']}")
        print(f"  说明: {config['description']}")
        
        if config['mode'] == "rgb_only":
            print("  ✓ 推荐：训练初期使用，速度快")
        elif config['mode'] == "event_only":
            print("  ⚠️ 注意：依赖完整的 Event 数据集")
        elif config['mode'] == "dual_avg":
            print("  ⚠️ 注意：需要模型支持双模态融合")


if __name__ == "__main__":
    print("=" * 60)
    print("FusionCocoEvalCallback 使用示例")
    print("=" * 60)
    
    # 运行示例
    test_fusion_modes()