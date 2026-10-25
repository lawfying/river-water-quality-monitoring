# -*- coding: utf-8 -*-
"""
水面漂浮物目标检测模型训练 (YOLOv8) - RTX 5070 优化版
"""
import os
import torch
from ultralytics import YOLO

# 自动检测 GPU
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f}GB)")
else:
    print("[OK] 设备: CPU (无GPU)")

# RTX 5070 12GB 推荐配置（640分辨率下 batch 降至 8）
BATCH = 8 if torch.cuda.is_available() else 4
MODEL_SIZE = 'm' if torch.cuda.is_available() else 'n'
print(f"   Batch: {BATCH}, 模型: YOLOv8{MODEL_SIZE}")


def train_yolo(data_yaml, epochs=100, imgsz=640, batch=None, model_size=None):
    batch = batch or BATCH
    model_size = model_size or MODEL_SIZE
    
    # 统计增强后有效数据量
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    train_img_dir = data_cfg.get('train', '')
    num_originals = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg','.jpeg','.png'))]) if os.path.exists(train_img_dir) else 0
    
    # 每轮增强组合数估算
    mosaic_factor = 4.0              # mosaic=1.0: 4张拼1张
    mixup_factor = 1.0 + 0.2 * 2.0  # mixup=0.2: 线性混合
    flip_factor = 1.0 + 0.5 + 0.1   # fliplr=0.5 + flipud=0.1
    geo_factor = 1.5                # degrees/translate/scale/shear 连续变换
    color_factor = 2.0              # HSV 连续抖动
    copy_paste_factor = 1.0 + 0.1 * 2.0  # copy_paste=0.1
    effective_multiplier = mosaic_factor * mixup_factor * flip_factor * geo_factor * color_factor * copy_paste_factor
    effective_per_epoch = num_originals * effective_multiplier
    total_effective = effective_per_epoch * epochs
    
    print(f"\n{'='*50}")
    print(f"  数据增强效果估算")
    print(f"{'='*50}")
    print(f"  原始训练图片:         {num_originals} 张")
    print(f"  增强组合倍率:         {effective_multiplier:.1f}x")
    print(f"  每轮有效样本量:       {effective_per_epoch:.0f} 张")
    print(f"  {epochs}轮总有效样本量:    {total_effective:.0f} 张")
    print(f"  相当于原始数据量:     {total_effective/max(num_originals,1):.0f}x")
    print(f"{'='*50}\n")
    
    print(f"\n开始训练 YOLOv8{model_size}")
    print(f"   数据: {data_yaml}")
    print(f"   设备: {DEVICE}, 尺寸: {imgsz}x{imgsz}, Batch: {batch}, Epochs: {epochs}")
    
    model = YOLO(f'yolo/yolov8{model_size}.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='output',
        name='float_detect',
        exist_ok=True,
        patience=20,
        save=True,
        device=DEVICE,
        augment=True,
        # ---- 数据增强参数 ----
        mosaic=1.0,           # 马赛克拼接（4图合1）
        mixup=0.2,            # 混合增强（从0.1提至0.2）
        copy_paste=0.1,       # 复制粘贴增强，小目标效果显著
        fliplr=0.5,           # 水平翻转（50%概率）
        flipud=0.1,           # 垂直翻转（10%概率），水面目标方向多样
        hsv_h=0.015,          # 色调抖动
        hsv_s=0.7,            # 饱和度抖动，模拟不同光照条件
        hsv_v=0.4,            # 明度抖动，模拟阴天/强光
        degrees=10.0,         # 旋转±10度，模拟拍摄角度变化
        translate=0.1,        # 平移10%
        scale=0.5,            # 缩放±50%，模拟不同拍摄距离
        shear=2.0,            # 剪切±2度
        workers=0,      # 单进程加载，防止页面文件溢出
        lr0=0.01,
        lrf=0.01,
        verbose=True
    )
    
    best_path = os.path.join('output', 'float_detect', 'weights', 'best.pt')
    print(f"\n[完成] 训练完成！最优模型: {best_path}")
    return results


def validate_model(model_path, data_yaml):
    print(f"\n[图表] 验证模型: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=DEVICE)
    print(f"\n[结果] 验证结果:")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP75:    {metrics.box.map75:.4f}")
    return metrics


if __name__ == "__main__":
    data_yaml = r'C:\water-llm\data\float_images\data.yaml'
    
    if not os.path.exists(data_yaml):
        print(f"[错误] 未找到 {data_yaml}")
        print("   请先运行: python src_float/auto_prepare.py")
        exit(1)
    
    # 开始训练（优化版：640分辨率 + 数据增强）
    train_yolo(data_yaml, epochs=100, imgsz=640)
    
    # 验证
    best_model = os.path.join('output', 'float_detect', 'weights', 'best.pt')
    if os.path.exists(best_model):
        validate_model(best_model, data_yaml)