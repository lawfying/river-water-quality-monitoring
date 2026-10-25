# -*- coding: utf-8 -*-
"""
水面漂浮物数据集自动准备脚本（全类别版）
VOC XML → YOLO txt → 自动划分 train/val/test → 生成 data.yaml
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
import yaml

random.seed(42)

# ========== 路径配置 ==========
VOC_ANNOTATIONS = r"C:\water-llm\Waste Detection Dataset\Annotations"
VOC_IMAGES = r"C:\water-llm\Waste Detection Dataset\JPEGImages"
OUTPUT_BASE = r"C:\water-llm\data\float_images"

# 13 个统一后的类别（按数量排序）
CLASS_NAMES = [
    'branch',      # 2111
    'bottle',      # 1486
    'leaf',        # 1190
    'plastic_bag', # 826 (823+3)
    'can',         # 746
    'milk_box',    # 543 (542+1)
    'grass',       # 481
    'plastic_box', # 309
    'bad_fruit',   # 248
    'glass_bottle',# 133 (128+5)
    'ball',        # 37
    'paper',       # 2
    'garbage',     # 1
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# TEST = 0.1


def convert_single_xml(xml_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    
    yolo_lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        if name not in class_names:
            continue
        
        cls_id = class_names.index(name)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        
        x_c = max(0, min(1, x_c))
        y_c = max(0, min(1, y_c))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    return yolo_lines


def create_dirs(base):
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(base, split, sub), exist_ok=True)


def main():
    print("🚀 全类别数据准备开始...")
    print(f"   标注: {VOC_ANNOTATIONS}")
    print(f"   图片: {VOC_IMAGES}")
    print(f"   输出: {OUTPUT_BASE}")
    print(f"   类别: {CLASS_NAMES}")
    
    create_dirs(OUTPUT_BASE)
    
    xml_files = sorted([f for f in os.listdir(VOC_ANNOTATIONS) if f.endswith('.xml')])
    total = len(xml_files)
    print(f"\n📦 找到 {total} 个标注文件")
    
    random.shuffle(xml_files)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    
    splits = {
        'train': xml_files[:n_train],
        'val': xml_files[n_train:n_train + n_val],
        'test': xml_files[n_train + n_val:]
    }
    
    print(f"📊 划分: 训练{n_train} | 验证{n_val} | 测试{total - n_train - n_val}")
    
    total_objs = 0
    skipped_files = []
    
    for split_name, files in splits.items():
        split_objs = 0
        for xml_file in files:
            base_name = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(VOC_ANNOTATIONS, xml_file)
            img_path = os.path.join(VOC_IMAGES, base_name + '.jpg')
            
            if not os.path.exists(img_path):
                skipped_files.append(f"{xml_file} (图片缺失)")
                continue
            
            yolo_lines = convert_single_xml(xml_path, CLASS_NAMES)
            if not yolo_lines:
                skipped_files.append(xml_file)
                continue
            
            txt_path = os.path.join(OUTPUT_BASE, split_name, 'labels', base_name + '.txt')
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            dst_img = os.path.join(OUTPUT_BASE, split_name, 'images', base_name + '.jpg')
            shutil.copy(img_path, dst_img)
            
            split_objs += len(yolo_lines)
        
        print(f"  ✅ {split_name}: {len(files)} 张, {split_objs} 个目标")
        total_objs += split_objs
    
    print(f"\n📦 总计: {total} 张图片, {total_objs} 个目标")
    if skipped_files:
        print(f"   ⚠️ 跳过 {len(skipped_files)} 个文件（无有效标注或图片缺失）")
    
    yaml_path = os.path.join(OUTPUT_BASE, 'data.yaml')
    data = {
        'path': OUTPUT_BASE.replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    
    print(f"\n✅ data.yaml 已生成: {yaml_path}")
    print(f"   nc: {len(CLASS_NAMES)}, names: {CLASS_NAMES}")
    print("\n🎉 运行: python src_float/train.py")


if __name__ == "__main__":
    main()