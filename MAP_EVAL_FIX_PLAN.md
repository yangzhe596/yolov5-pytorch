"""
 Fusion mAP 评估修复方案

 问题分析:
 1. JSON 中的 file_name 格式: "0/PADDED_RGB/Video_0_16_03_17.465070.jpg"
 2. 图片实际位置: datasets/fred_coco/rgb/val/0/PADDED_RGB/Video_0_16_03_17.465070.jpg
 3. 但文件夹可能不存在或路径不匹配

 解决方案:
 修复 FusionCocoEvalCallback 的图片路径处理逻辑
"""

print("""
================================================================
Fusion mAP 评估修复方案
================================================================

问题:
- mAP 一直是 0
- 评估代码找不到图片文件
- file_name 格式为 "0/PADDED_RGB/Video_0_16_03_17.465070.jpg"

原因:
- JSON 文件中的 file_name 包含子目录路径
- 实际图片位置与 JSON 中的 file_name 不匹配

解决方案:
================================================================

方案 1: 修复 train_fred_fusion.py 中的路径处理（推荐）
----------------------------------------------------------------

在 train_fred_fusion.py 的 FusionCocoEvalCallback 类中修复图片路径处理：

步骤 1: 修改 generate_result_files 方法

找到这一行：
    img_path = os.path.join(self.image_dir, file_name)

替换为：
    # 处理子目录路径
    full_path = os.path.join(self.image_dir, file_name)
    simple_path = os.path.join(self.image_dir, os.path.basename(file_name))
    
    if os.path.exists(full_path):
        img_path = full_path
    elif os.path.exists(simple_path):
        img_path = simple_path
        print(f"⚠️  路径修复: {file_name} -> {os.path.basename(file_name)}")
    else:
        # 如果都不存在，尝试查找图片文件
        img_name = os.path.splitext(os.path.basename(file_name))[0]
        pattern = os.path.join(self.image_dir, f"**/*{img_name}.*")
        import glob
        matches = glob.glob(pattern, recursive=True)
        if matches:
            img_path = matches[0]
            print(f"✓  找到图片: {matches[0]}")
        else:
            print(f"✗  图片不存在: {file_name}")
            pbar.update(1)
            continue

步骤 2: 在 get_map_txt 中也添加类似修复（可选）

方案 2: 创建图片软链接
----------------------------------------------------------------

创建软链接将 JSON 中的路径指向实际图片位置：

运行创建链接脚本：
    python create_image_symlinks.py

方案 3: 修改 data_loader 以支持路径映射
----------------------------------------------------------------

在 Fusion 数据加载器中添加路径映射逻辑

================================================================
运行修复命令:
================================================================

""")