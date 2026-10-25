# -*- coding: utf-8 -*-
"""
水质监测数值数据特征提取与分析
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

print("1. 正在读取清洗后的数据...")
try:
    df = pd.read_csv('data/water_potability_clean.csv')
    print(f"数据读取成功，当前形状: {df.shape}")
except FileNotFoundError:
    print("错误：未找到 data/water_potability_clean.csv，请先运行 data_cleaning.py")
    exit()

# --- 2. 相关性热力图 ---
print("2. 正在生成相关性热力图...")
# 【核心修复】：使用 select_dtypes 只提取数值类型的列，自动过滤掉类似 'Ws-SY-03' 的字符串ID
numeric_df = df.select_dtypes(include=['number'])

# 计算相关系数矩阵
correlation_matrix = numeric_df.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('水质指标相关系数矩阵热力图')
plt.tight_layout()
plt.savefig('output/heatmap_correlation.png')
plt.show()
print("热力图已保存至 output/heatmap_correlation.png")

# --- 3. 特征分布图 ---
print("3. 正在生成特征分布图...")
numeric_features = numeric_df.columns.tolist()

# 创建子图 (3行3列)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_features[:9]):
    if idx < len(axes):
        ax = axes[idx]
        sns.histplot(numeric_df[col], kde=True, ax=ax, color='skyblue')
        ax.set_title(f'{col} 分布')

plt.suptitle('主要水质特征分布直方图')
plt.tight_layout()
plt.savefig('output/feature_histograms.png')
plt.show()
print("特征分布图已保存至 output/feature_histograms.png")

# --- 4. 标签与特征关系 (Pairplot) ---
print("4. 正在生成标签与特征关系图 (Pairplot)...")
# 采样防止数据量过大导致画图过慢或内存溢出
sample_df = numeric_df.sample(n=min(500, len(numeric_df)), random_state=42)

# 如果原数据有 Potability 标签，也把它加到采样数据里用于着色
if 'Potability' in df.columns:
    sample_df['Potability'] = df.loc[sample_df.index, 'Potability']
    g = sns.pairplot(sample_df, hue='Potability', palette='Set1', plot_kws={'s': 15})
else:
    g = sns.pairplot(sample_df, palette='Set1', plot_kws={'s': 15})

g.fig.suptitle('水质标签与特征关系散点图矩阵', y=1.02)
plt.savefig('output/pairplot_relationship.png')
plt.show()
print("关系散点图已保存至 output/pairplot_relationship.png")

print("\n[OK] 特征工程分析全部完成！请查看 output 目录下的图片。")