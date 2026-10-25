# -*- coding: utf-8 -*-
"""
水质监测数值数据获取与预处理 (增强修复版)
修复项目：
.1 [逻辑增强] 引入双重兜底机制：在分组填充后增加全局均值校验，确保 PH 值缺失率降至 0%。
.2[性能优化] 重构为向量化布尔索引操作，解决 Pandas apply 赋值失效问题并提升效率。
.3[工程规范] 补全 os 模块导入，增加过程日志输出，增强代码鲁棒性。
.4[新增] 支持同时清洗 water_potability.csv 和 water_potability_2026.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_water_data(input_path, output_path, dataset_label=""):
    """通用的水质数据清洗函数"""
    print(f"\n{'='*50}")
    print(f"正在清洗: {input_path} {dataset_label}")
    print(f"{'='*50}")

    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值情况:\n{df.isnull().sum()}")

    # 2. 增强版 PH值缺失值处理
    print(f"\n2. 正在处理 PH 值缺失 (增强逻辑)...")

    # --- 第一步：尝试按文档逻辑分组填充 ---
    cond_1_mean = df[(df['Potability'] == 0) & (df['Hardness'] <= 150)]['ph'].mean()
    cond_2_mean = df[(df['Potability'] == 0) & (df['Hardness'] > 150)]['ph'].mean()
    cond_3_mean = df[(df['Potability'] == 1) & (df['Hardness'] <= 150)]['ph'].mean()
    cond_4_mean = df[(df['Potability'] == 1) & (df['Hardness'] > 150)]['ph'].mean()

    print(f"   [调试] 合格软水PH均值: {cond_1_mean:.2f}")
    print(f"   [调试] 合格硬水PH均值: {cond_2_mean:.2f}")
    print(f"   [调试] 不合格软水PH均值: {cond_3_mean:.2f}")
    print(f"   [调试] 不合格硬水PH均值: {cond_4_mean:.2f}")

    df.loc[(df['ph'].isna()) & (df['Potability'] == 0) & (df['Hardness'] <= 150), 'ph'] = cond_1_mean
    df.loc[(df['ph'].isna()) & (df['Potability'] == 0) & (df['Hardness'] > 150), 'ph'] = cond_2_mean
    df.loc[(df['ph'].isna()) & (df['Potability'] == 1) & (df['Hardness'] <= 150), 'ph'] = cond_3_mean
    df.loc[(df['ph'].isna()) & (df['Potability'] == 1) & (df['Hardness'] > 150), 'ph'] = cond_4_mean

    remaining_ph_na = df['ph'].isna().sum()
    if remaining_ph_na > 0:
        global_mean = df['ph'].mean()
        print(f"   警告: 仍有 {remaining_ph_na} 个PH值未被分组逻辑覆盖，正在使用全局均值 {global_mean:.2f} 进行兜底填充...")
        df['ph'] = df['ph'].fillna(global_mean)

    # 3. 硫酸盐 (Sulfate) 缺失值处理
    print("3. 正在处理 Sulfate 缺失...")
    mean_sulfate_0 = df[df['Potability'] == 0]['Sulfate'].mean()
    mean_sulfate_1 = df[df['Potability'] == 1]['Sulfate'].mean()
    df.loc[(df['Sulfate'].isna()) & (df['Potability'] == 0), 'Sulfate'] = mean_sulfate_0
    df.loc[(df['Sulfate'].isna()) & (df['Potability'] == 1), 'Sulfate'] = mean_sulfate_1

    # 4. 三卤甲烷 (Trihalomethanes) 缺失值处理
    print("4. 正在处理 Trihalomethanes 缺失 (删除行)...")
    df = df.dropna(subset=['Trihalomethanes'])

    # 5. 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n== 清洗完成！保存至 {output_path}")
    print(f"   清洗后数据形状: {df.shape}")
    print(f"   缺失值检查:\n{df.isnull().sum()}")

    return df


if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)

    # 清洗原始数据集
    clean_water_data(
        input_path='data/water_potability.csv',
        output_path='data/water_potability_clean.csv',
        dataset_label="(历史数据 2019-2024)"
    )

    # 清洗2026年新数据集
    clean_water_data(
        input_path='data/water_potability_2026.csv',
        output_path='data/water_potability_2026_clean.csv',
        dataset_label="(2026年新数据)"
    )

    print(f"\n{'='*50}")
    print("全部数据清洗完成！")
    print(f"{'='*50}")
