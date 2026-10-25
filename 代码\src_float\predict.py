# -*- coding: utf-8 -*-
"""
水面漂浮物检测推理脚本（13类别版）
"""
import os
from ultralytics import YOLO

CLASS_NAMES = [
    'branch', 'bottle', 'leaf', 'plastic_bag', 'can', 'milk_box',
    'grass', 'plastic_box', 'bad_fruit', 'glass_bottle', 'ball',
    'paper', 'garbage'
]

def predict_image(model_path, image_path, conf=0.25, save=True):
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return None
    
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, save=save,
                           project='output', name='float_predict')
    
    detections = []
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个目标:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            detections.append({
                'class': class_name,
                'confidence': round(conf_score, 4),
                'bbox': [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })
            print(f"  → {class_name}: {conf_score:.2%}")
    
    return detections


def batch_predict(model_path, image_dir, conf=0.25):
    model = YOLO(model_path)
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("❌ 未找到测试图像")
        return []
    
    results = model.predict(image_files, conf=conf, save=True,
                           project='output', name='float_batch')
    
    total = sum(len(r.boxes) for r in results)
    print(f"\n✅ 批量检测完成: {len(image_files)} 张图, {total} 个目标")
    
    # 统计各类别数量
    from collections import Counter
    all_classes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            all_classes.append(CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
    
    print("\n📊 类别分布:")
    for cls, count in Counter(all_classes).most_common():
        print(f"   {cls}: {count}")
    
    return results


if __name__ == "__main__":
    model_path = r'C:\water-llm\output\float_detect\weights\best.pt'
    
    if not os.path.exists(model_path):
        model_path = 'yolov8s.pt'
        print("⚠️ 未找到训练模型，使用预训练权重演示")
    
    # 单张测试
    test_img = r'C:\water-llm\data\float_images\test\images\000001.jpg'
    if os.path.exists(test_img):
        predict_image(model_path, test_img)
    else:
        print(f"⚠️ 测试图像不存在: {test_img}")