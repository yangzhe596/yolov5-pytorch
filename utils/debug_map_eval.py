"""
Fusion mAP 评估调试工具
检查 mAP 为 0 的原因
"""
import os
import json
import sys
from PIL import Image

sys.path.insert(0, '/mnt/data/code/yolov5-pytorch')

def check_coco_json(json_path):
    """检查 COCO JSON 文件"""
    print("="*60)
    print("检查 COCO JSON 文件")
    print("="*60)
    print(f"JSON 路径: {json_path}")
    
    if not os.path.exists(json_path):
        print("❌ JSON 文件不存在")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n✓ JSON 文件存在")
    print(f"  - 图片数量: {len(data.get('images', []))}")
    print(f"  - 标注数量: {len(data.get('annotations', []))}")
    print(f"  - 类别数量: {len(data.get('categories', []))}")
    
    if data.get('categories'):
        print(f"\n类别列表:")
        for cat in data['categories']:
            print(f"  - ID: {cat['id']}, 名称: {cat['name']}")
    
    # 检查是否有图片和标注
    if len(data.get('images', [])) == 0:
        print("\n❌ 警告: 没有图片记录")
        return False
    
    if len(data.get('annotations', [])) == 0:
        print("\n❌ 警告: 没有标注记录")
        return False
    
    print("\n✅ JSON 文件格式正确")
    return data


def check_image_paths(json_data, image_dir):
    """检查图片路径是否存在"""
    print("\n" + "="*60)
    print(f"检查图片路径")
    print("="*60)
    print(f"图片目录: {image_dir}")
    
    images = json_data.get('images', [])
    total = len(images)
    found = 0
    
    # 检查前 10 张图片
    sample_count = min(10, total)
    print(f"\n检查前 {sample_count} 张图片...")
    
    for i, img_info in enumerate(images[:sample_count]):
        file_name = img_info.get('file_name', '')
        img_id = img_info.get('id', '')
        
        # 构建完整路径
        full_path = os.path.join(image_dir, file_name)
        
        exists = os.path.exists(full_path)
        if exists:
            found += 1
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name}")
    
    # 检查所有图片路径（只统计，不打印）
    print(f"\n统计所有图片路径...")
    for img_info in images:
        file_name = img_info.get('file_name', '')
        full_path = os.path.join(image_dir, file_name)
        if os.path.exists(full_path):
            found += 1
    
    print(f"  - 总图片数: {total}")
    print(f"  - 存在图片: {found}/{total}")
    print(f"  - 缺失图片: {total - found}/{total}")
    
    if found == 0:
        print("\n❌ 错误: 所有图片都找不到")
        print("\n可能的原因:")
        print(f"  1. image_dir 设置错误")
        print(f"  2. 图片文件格式不匹配（可能是 .png 而不是 .jpg）")
        print(f"  3. JSON 中的 file_name 与实际文件名不匹配")
        return False
    
    missing_ratio = (total - found) / total
    if missing_ratio > 0.5:
        print(f"\n⚠️ 警告: 超过 50% 的图片找不到")
        return False
    
    print("\n✅ 图片路径检查通过")
    return found


def check_annotations(json_data, image_dir):
    """检查标注是否合理"""
    print("\n" + "="*60)
    print("检查标注数据")
    print("="*60)
    
    annotations = json_data.get('annotations', [])
    images = json_data.get('images', [])
    
    if not annotations:
        print("❌ 没有标注数据")
        return False
    
    print(f"标注总数: {len(annotations)}")
    
    # 检查标注范围
    bbox_stats = {'valid': 0, 'empty': 0, 'invalid': 0}
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            bbox_stats['invalid'] += 1
            continue
        
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            bbox_stats['empty'] += 1
        else:
            bbox_stats['valid'] += 1
    
    print(f"\n边界框统计:")
    print(f"  - 有效边界框: {bbox_stats['valid']}")
    print(f"  - 空边界框: {bbox_stats['empty']}")
    print(f"  - 非法边界框: {bbox_stats['invalid']}")
    
    if bbox_stats['valid'] == 0:
        print("\n❌ 错误: 没有有效边界框")
        return False
    
    # 检查标注的图片是否存在
    img_ids_with_ann = set(ann['image_id'] for ann in annotations)
    img_ids_all = set(img['id'] for img in images)
    
    matched = len(img_ids_with_ann & img_ids_all)
    total = len(img_ids_with_ann)
    
    print(f"\n标注匹配统计:")
    print(f"  - 有标注的图片: {total}")
    print(f"  - 匹配的图片: {matched}/{total}")
    
    if matched == 0:
        print("\n❌ 错误: 没有匹配的图片 ID")
        return False
    
    print(f"\n✅ 标注检查通过")
    return True


def test_prediction(model_path, json_path, image_dir):
    """测试单张图片预测"""
    print("\n" + "="*60)
    print(f"测试单张图片预测")
    print("="*60)
    
    # 简单的路径检查
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("跳过预测测试")
        return
    
    print("预测测试需要更多时间...")
    print("请确保:")
    print("  1. 模型文件存在")
    print("  2. PyTorch 环境正常")
    print("  3. 有 GPU 可用")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Fusion mAP 评估调试工具")
    print("="*60)
    
    # 配置路径
    base_dir = '/mnt/data/code/yolov5-pytorch'
    json_path = 'datasets/fred_coco/rgb/annotations/instances_val.json'
    image_dirs = [
        'datasets/fred_coco/rgb/val',      # 推荐路径
        'datasets/fred_coco/rgb/images',   # 备用路径
    ]
    
    log_dir = 'logs/fred_rgb'
    model_path = os.path.join(log_dir, 'fred_rgb_best.pth')
    
    # 步骤 1: 检查 JSON 文件
    print("\n步骤 1: 检查 COCO JSON 文件")
    json_data = check_coco_json(json_path)
    
    if not json_data:
        print("\n❌ JSON 文件有问题，无法继续")
        return
    
    # 步骤 2: 尝试查找图片目录
    print("\n\n步骤 2: 尝试查找图片目录")
    valid_image_dir = None
    
    for img_dir in image_dirs:
        print(f"\n尝试: {img_dir}")
        if check_image_paths(json_data, img_dir):
            valid_image_dir = img_dir
            break
    
    if not valid_image_dir:
        print("\n❌ 所有图片目录都找不到图片")
        print("\n建议:")
        print(f"  - 检查图片实际位置")
        print(f"  - 检查 JSON 中的 file_name")
        print("  - 检查文件格式（.jpg/.png）")
        return
    
    # 步骤 3: 检查标注
    print("\n\n步骤 3: 检查标注数据")
    if not check_annotations(json_data, valid_image_dir):
        return
    
    # 步骤 4: 测试预测（可选）
    test_prediction(model_path, json_path, valid_image_dir)
    
    # 总结
    print("\n" + "="*60)
    print("调试总结")
    print("="*60)
    print(f"推荐的配置:")
    print(f"  JSON 路径: {json_path}")
    print(f"  图片目录: {valid_image_dir}")
    print(f"  模型路径: {model_path if os.path.exists(model_path) else '(训练后生成)'}")
    print("\n")

    # 创建配置示例
    print("在 train_fred_fusion.py FusionCocoEvalCallback 中使用:")
    print("-" * 60)
    print(f"""
eval_callback = FusionCocoEvalCallback(
    # ... 其他参数
    coco_json_path='{json_path}',
    image_dir='{valid_image_dir}',  # ✅ 使用正确的路径
    log_dir='{log_dir}',
    # ... 其他参数
)
    """)


if __name__ == "__main__":
    main()