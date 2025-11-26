"""
FusionCocoEvalCallback 图片路径修复

自动修复 train_fred_fusion.py 中的图片路径加载逻辑
"""
import shutil
import os

def fix_eval_callback_image_loading():
    """
    修复 FusionCocoEvalCallback 的图片路径加载问题
    """
    train_file = '/mnt/data/code/yolov5-pytorch/train_fred_fusion.py'
    backup_file = '/mnt/data/code/yolov5-pytorch/train_fred_fusion.py.backup'
    
    # 备份原文件
    if not os.path.exists(backup_file):
        shutil.copy(train_file, backup_file)
        print(f"✓ 已备份原文件: {backup_file}")
    
    # 读取文件
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("="*60)
    print("修复图片路径加载逻辑")
    print("="*60)
    
    # 查找需要修复的位置
    old_code_start1 = content.find('img_path = os.path.join(self.image_dir, file_name)')
    old_code_start2 = content.find('img_path = os.path.join(self.log_dir, file_name.replace')
    
    if old_code_start1 == -1:
        print("⚠️  未找到目标代码 1")
        return False
    
    # 找到完整的代码块
    # 我们需要替换的是 generate_result_files 中的图片路径处理
    
    # 方案：添加一个辅助方法来处理路径，然后调用它
    helper_method = '''
    def _find_image_path(self, file_name: str) -> str:
        """
        安全地查找图片路径，处理 JSON 中的子目录路径
        
        Args:
            file_name: COCO JSON 中的 file_name (可能包含子目录)
            
        Returns:
            完整的图片路径（如果找到），否则返回空字符串
        """
        # 方案 1: 直接使用完整路径
        full_path = os.path.join(self.image_dir, file_name)
        if os.path.exists(full_path):
            return full_path
        
        # 方案 2: 只使用文件名（去掉子目录）
        simple_name = os.path.basename(file_name)
        simple_path = os.path.join(self.image_dir, simple_name)
        if os.path.exists(simple_path):
            # 记录路径映射，避免重复映射
            if not hasattr(self, 'path_mapping'):
                self.path_mapping = {}
            self.path_mapping[file_name] = simple_path
            return simple_path
        
        # 方案 3: 在整个目录中查找（慢，但可能会找到）
        import glob
        img_name_no_ext = os.path.splitext(simple_name)[0]
        pattern = os.path.join(self.image_dir, f"**/*{img_name_no_ext}.*")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            found_path = matches[0]
            print(f"  ⚠️  查找到图片: {file_name} -> {found_path}")
            return found_path
        
        # 方案 4: 检查 zipped 子目录
        zipped_path = os.path.join(self.image_dir, "zipped", file_name)
        if os.path.exists(zipped_path):
            return zipped_path
        
        # 方案 5: 检查 RGB 子目录
        rgb_path = os.path.join(self.image_dir, "RGB", file_name)
        if os.path.exists(rgb_path):
            return rgb_path
        
        # 方案 6: 检查是否存在同名文件（忽略子目录）
        base_name = os.path.basename(file_name)
        candidates = [
            os.path.join(self.image_dir, base_name),
            os.path.join(self.image_dir, "val", base_name),
            os.path.join(self.image_dir, "train", base_name),
            os.path.join(self.image_dir, "test", base_name),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        
        return ""  # 真的找不到
    
'''
    
    # 插入辅助方法（在类的开头添加）
    insert_pos = content.find('class FusionCocoEvalCallback:') + len('class FusionCocoEvalCallback:')
    # 找到 __init__ 开始的位置，在它之前插入
    init_pos = content.find('def __init__(', insert_pos)
    
    if init_pos == -1:
        print("⚠️  未找到 __init__ 方法")
        return False
    
    # 在 __init__ 之前插入辅助方法
    new_content = content[:init_pos] + helper_method + '\n' + content[init_pos:]
    
    # 替换图片路径加载代码
    old_pattern = '''                image_id = os.path.splitext(os.path.basename(file_name))[0]
                img_path = os.path.join(self.image_dir, file_name)
                
                if not os.path.exists(img_path):
                    alt_path = os.path.join(self.log_dir, file_name.replace("datasets/fred_coco/", ""))
                    if not os.path.exists(alt_path):
                        pbar.update(1)
                        continue
                    img_path = alt_path'''
    
    new_pattern = '''                image_id = os.path.splitext(os.path.basename(file_name))[0]
                img_path = self._find_image_path(file_name)
                
                if not img_path or not os.path.exists(img_path):
                    # 尝试相对路径
                    alt_path = os.path.join(self.log_dir, file_name.replace("datasets/fred_coco/", ""))
                    if os.path.exists(alt_path):
                        img_path = alt_path
                    else:
                        print(f"  ✗ 找不到图片: {file_name}")
                        pbar.update(1)
                        continue'''
    
    if old_pattern in new_content:
        new_content = new_content.replace(old_pattern, new_pattern)
        print("✓ 已修复 generate_result_files 方法")
    else:
        print("⚠️  未找到 generate_result_files 的图片加载代码")
        print("   尝试查找其他相似代码...")
        
        # 尝试更宽松的匹配
        import re
        pattern = re.compile(
            r'img_path = os\.path\.join\(self\.image_dir, file_name\)([\s\S]{100,200}?)(if not os\.path\.exists\(img_path\):)',
            re.MULTILINE
        )
        
        matches = list(pattern.finditer(new_content))
        if matches:
            # 直接替换整个块
            replacement = '''                img_path = self._find_image_path(file_name)
                
                if not img_path or not os.path.exists(img_path):
                    # 尝试相对路径
                    alt_path = os.path.join(self.log_dir, file_name.replace("datasets/fred_coco/", ""))
                    if os.path.exists(alt_path):
                        img_path = alt_path
                    else:
                        print(f"  ✗ 找不到图片: {file_name}")
                        pbar.update(1)
                        continue'''
            
            # 获取第一个匹配的位置
            match = matches[0]
            # 找到完整的代码块
            old_code = new_content[match.start():match.start() + 300]
            
            # 替换
            new_content = new_content.replace(old_code, replacement)
            print("✓ 已修复（使用宽松匹配）")
        else:
            print("⚠️  警告：可能修复不完全，请手动检查")
    
    # 写入修改后的内容
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print("✓ 已备份原文件:", backup_file)
    print("✓ 已添加图片路径查找辅助方法")
    print("✓ 已修复 generate_result_files 中的图片加载逻辑")
    print("\n现在可以重新运行训练:")
    print("  python train_fred_fusion.py --modality rgb")
    
    return True


def create_debug_version():
    """
    创建一个调试版本的 FusionCocoEvalCallback
    用于测试图片路径设置是否正确
    """
    debug_file = '/mnt/data/code/yolov5-pytorch/utils/debug_eval_callback.py'
    
    code = '''
"""
FusionCocoEvalCallback 调试版本

用于测试图片路径设置是否正确
"""
import os
import json

def debug_fusion_eval(json_path, image_dir, test_mode="rgb_only"):
    """调试 Fusion 评估的图片路径"""
    
    print("="*60)
    print("Fusion 评估调试")
    print("="*60)
    
    # 1. 检查 JSON
    print(f"\\n1. 检查 COCO JSON: {json_path}")
    if not os.path.exists(json_path):
        print(f"  ✗ JSON 不存在")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  ✓ JSON 存在")
    print(f"    - 图片: {len(data.get('images', []))}")
    print(f"    - 标注: {len(data.get('annotations', []))}")
    
    # 2. 检查图片目录
    print(f"\\n2. 检查图片目录: {image_dir}")
    if test_mode == "rgb_only":
        test_dirs = [image_dir, os.path.join(image_dir, "rgb"), os.path.join(image_dir, "val")]
    else:
        test_dirs = [image_dir]
    
    valid_dir = None
    for d in test_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            print(f"  ✓ 找到目录: {d}")
            valid_dir = d
            break
    
    if not valid_dir:
        print(f"  ✗ 没有有效的图片目录")
        return False
    
    # 3. 测试图片路径映射
    print(f"\\n3. 测试图片路径映射")
    images = data.get('images', [])[:10]  # 只测试前 10 张
    
    found_count = 0
    for img_info in images:
        file_name = img_info.get('file_name', '')
        img_id = img_info.get('id', '')
        print(f"\\n  测试: {file_name}")
        
        # 方法 1: 完整路径
        full_path = os.path.join(valid_dir, file_name)
        if os.path.exists(full_path):
            print(f"    ✓ 完整路径: {full_path}")
            found_count += 1
            continue
        
        # 方法 2: 去掉子目录
        simple_name = os.path.basename(file_name)
        simple_path = os.path.join(valid_dir, simple_name)
        if os.path.exists(simple_path):
            print(f"    ✓ 简化路径: {simple_path}")
            found_count += 1
            continue
        
        # 方法 3: 在子目录中查找
        img_base = os.path.splitext(simple_name)[0]
        subdir_pattern = os.path.join(valid_dir, "*", simple_name)
        import glob
        matches = glob.glob(subdir_pattern)
        if matches:
            print(f"    ✓ 匹配路径: {matches[0]}")
            found_count += 1
            continue
        
        # 方法 4: 遍历查找
        matches = []
        for root, dirs, files in os.walk(valid_dir):
            for f in files:
                if f == simple_name or img_base in f:
                    matches.append(os.path.join(root, f))
                    break
            if matches:
                break
        
        if matches:
            print(f"    ✓ 遍历找到: {matches[0]}")
            found_count += 1
            continue
        
        print(f"    ✗ 未找到")
    
    print(f"\\n4. 统计结果")
    print(f"  - 测试图片: {len(images)}")
    print(f"  - 找到图片: {found_count}/{len(images)}")
    
    if found_count == 0:
        print(f"\\n  ✗ 错误: 没有找到任何图片")
        print(f"\\n  建议:")
        print(f"    1. 检查图片是否实际存在")
        print(f"    2. 检查目录结构")
        print(f"    3. 临时修改 JSON 文件中的 file_name")
        return False
    else:
        print(f"\\n  ✓ 图片路径设置正确")
        return True


if __name__ == "__main__":
    # 配置
    json_path = "datasets/fred_coco/rgb/annotations/instances_val.json"
    image_dir = "datasets/fred_coco/rgb"
    
    debug_fusion_eval(json_path, image_dir, "rgb_only")
'''
    
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"\\n✓ 已创建调试脚本: {debug_file}")
    print("  运行: python utils/debug_eval_callback.py")


def main():
    """主函数"""
    
    print("""
   ==================================================================
                          Fusion mAP 评估修复
   ==================================================================
    
   检测到 mAP 为 0 的问题，原因可能包括:
    
   1. 图片路径不正确
   2. JSON 中的 file_name 与实际文件不匹配
   3. 评估代码路径处理逻辑有问题
    
   这将自动修复 train_fred_fusion.py 中的图片加载逻辑。
    
   ==================================================================
    
""")
    
    # 创建调试脚本
    create_debug_version()
    
    print("\n\n" + "="*60)
    
    # 询问是否修复
    answer = input("是否立即修复路径加载代码？(y/n): ").strip().lower()
    
    if answer == 'y':
        success = fix_eval_callback_image_loading()
        if success:
            print("\\n\\n修复成功！请重新运行训练。")
        else:
            print("\\n\\n修复失败，请手动检查。")
    else:
        print("\\n跳过自动修复。")
        print("你可以手动修改 train_fred_fusion.py 或运行调试脚本。")

if __name__ == "__main__":
    main()