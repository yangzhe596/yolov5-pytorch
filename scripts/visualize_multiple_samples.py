#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化多张样本，验证coordinates.txt标注是否正确
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

# 选择序列5
seq_dir = Path('/mnt/data/datasets/fred/5')
rgb_dir = seq_dir / 'PADDED_RGB'

# 获取所有图像
images = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])
total_images = len(images)

print(f"序列5总图像数: {total_images}")

# 读取coordinates.txt
with open(seq_dir / 'coordinates.txt', 'r') as f:
    lines = f.readlines()

# 获取第一张图像时间
first_img = images[0]
parts = first_img.name.replace('.jpg', '').split('_')
first_abs_time = int(parts[2]) * 3600 + int(parts[3]) * 60 + float(parts[4])

# 获取第一个标注时间
first_ann_time = float(lines[0].split(':')[0])

# 计算视频起始时间
video_start = first_abs_time - first_ann_time

print(f"视频起始时间: {video_start:.6f} 秒")

# 选择多张图像进行测试
# 选择开始、1/4、1/2、3/4、结束位置的图像
test_indices = [
    0,                          # 开始
    total_images // 4,          # 1/4
    total_images // 2,          # 1/2
    total_images * 3 // 4,      # 3/4
    total_images - 1,           # 结束
]

# 再随机选择5张
random.seed(42)
random_indices = random.sample(range(total_images), 5)
test_indices.extend(random_indices)

print(f"\n测试 {len(test_indices)} 张图像...")

for idx, img_idx in enumerate(test_indices):
    img_path = images[img_idx]
    
    # 提取时间戳
    parts = img_path.name.replace('.jpg', '').split('_')
    hours = int(parts[2])
    minutes = int(parts[3])
    seconds = float(parts[4])
    abs_time = hours * 3600 + minutes * 60 + seconds
    rel_time = abs_time - video_start
    
    # 查找最接近的标注
    closest_line = None
    min_diff = float('inf')
    
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:
            ts = float(parts[0])
            diff = abs(ts - rel_time)
            if diff < min_diff:
                min_diff = diff
                closest_line = line.strip()
    
    # 解析bbox
    parts = closest_line.split(':')
    coords = [float(x.strip()) for x in parts[1].split(',')]
    x1, y1, x2, y2 = coords
    
    # 加载图像
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # 绘制bbox
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
    
    # 画中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    cross_size = 30
    draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y], fill='blue', width=3)
    draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size], fill='blue', width=3)
    
    # 添加文本标签
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = None
    
    text = f"#{img_idx} bbox: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) size:{x2-x1:.0f}x{y2-y1:.0f}"
    draw.text((10, 10), text, fill='yellow', font=font)
    
    text2 = f"time_diff: {min_diff:.4f}s"
    draw.text((10, 40), text2, fill='yellow', font=font)
    
    # 保存
    output_path = f'test_sample_{idx:02d}_idx{img_idx}.jpg'
    img.save(output_path)
    
    print(f"  [{idx+1}/{len(test_indices)}] {img_path.name}")
    print(f"      相对时间: {rel_time:.6f}s, 时间差: {min_diff:.6f}s")
    print(f"      bbox: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}), 尺寸: {x2-x1:.0f}x{y2-y1:.0f}")
    print(f"      保存: {output_path}")

print(f"\n✓ 完成！已生成 {len(test_indices)} 张测试图像")
print(f"文件名格式: test_sample_XX_idxYYYY.jpg")
print(f"\n请查看这些图片，确认红色框是否框住了目标")
