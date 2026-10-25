# -*- coding: utf-8 -*-
"""
水质识别分类模型训练与评估
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建输出目录
os.makedirs('output', exist_ok=True)
os.makedirs('configs', exist_ok=True)

print("1. 正在加载清洗后的数据（合并多个数据集）...")

datasets_to_load = [
    ('data/water_potability_clean.csv', '历史数据(2019-2024)'),
    ('data/water_potability_2026_clean.csv', '2026年新数据'),
]

df_list = []
for path, label in datasets_to_load:
    try:
        d = pd.read_csv(path)
        print(f"   [OK] {label}: {d.shape}")
        df_list.append(d)
    except FileNotFoundError:
        print(f"   [跳过] {label}: 未找到 {path}，跳过")

if not df_list:
    print("错误：未找到任何数据集，请先运行数据清洗脚本")
    exit()

df = pd.concat(df_list, ignore_index=True)
print(f"\n[OK] 合并后数据总形状: {df.shape}")
print(f"   各来源样本数: {[len(d) for d in df_list]}")

# --- 2. 数据预处理 ---
print("2. 正在进行数据预处理...")

# 1. 分离特征与标签
X = df.drop(columns=['Potability'], errors='ignore') # 移除标签
y = df['Potability']

# 2. 【关键修复】强制只保留数值列 (过滤掉站点编号等文本)
X = X.select_dtypes(include=['number'])

print(f"有效特征列: {X.columns.tolist()}")

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集大小: {X_train_scaled.shape}, 测试集大小: {X_test_scaled.shape}")

# --- SMOTE 过采样处理类别不平衡 ---
print("\n2.5 正在应用 SMOTE 过采样处理类别不平衡...")
print(f"   过采样前训练集分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"   过采样后训练集分布: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")
print(f"   训练集大小: {X_train_res.shape[0]} -> {len(X_train_res)}")

# --- 3. 定义评估函数 ---
def model_eval(model_name, model, X_train, X_test, y_train, y_test):
    """
    通用模型评估函数
    """
    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # 计算概率 (用于AUC)
    if hasattr(model, "predict_proba"):
        test_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_proba)
        print(f"{model_name} 测试集 AUC: {auc:.4f}")
    else:
        auc = None

    # 打印报告
    print(f'\n===== {model_name} 测试集分类报告 =====')
    print(classification_report(y_test, test_pred))

    # 绘制混淆矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练集混淆矩阵
    cm_train = confusion_matrix(y_train, train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{model_name} - 训练集混淆矩阵')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')

    # 测试集混淆矩阵
    cm_test = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title(f'{model_name} - 测试集混淆矩阵')
    ax2.set_xlabel('预测标签')
    ax2.set_ylabel('真实标签')

    plt.suptitle(f'{model_name} 模型评估结果')
    plt.tight_layout()
    plt.savefig(f'output/{model_name}_confusion_matrix.png')
    plt.show()

    return auc

# --- 4. 基础模型训练 ---
print("4. 正在训练基础模型...")

# 4.1 KNN
print("\n--- 训练 KNN 模型 ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_res, y_train_res)
model_eval('KNN', knn, X_train_res, X_test_scaled, y_train_res, y_test)

# 4.2 SVM
print("\n--- 训练 SVM 模型 ---")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_res, y_train_res)
model_eval('SVM', svm, X_train_res, X_test_scaled, y_train_res, y_test)

# 4.3 决策树
print("\n--- 训练 决策树 模型 ---")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
model_eval('DecisionTree', dt, X_train, X_test, y_train, y_test)

# --- 5. 随机森林网格搜索优化 ---
print("\n5. 正在对 随机森林 进行网格搜索优化...")

param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [15, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', None],
    'class_weight': ['balanced', None],
}

rf = RandomForestClassifier(random_state=42)
# 以AUC为优化目标
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n========== 随机森林最优超参数 (基于AUC) ==========")
print(grid_search.best_params_)
print(f"5折交叉验证最优AUC得分: {grid_search.best_score_:.4f}")

# --- 6. 评估最优模型 ---
best_rf = grid_search.best_estimator_
model_eval('Optimized_RandomForest', best_rf, X_train, X_test, y_train, y_test)

# --- 7. XGBoost 网格搜索优化 ---
print("\n7. 正在对 XGBoost 进行网格搜索优化...")

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2],
}

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)  # 树模型不需要缩放，使用原始特征

print("\n========== XGBoost 最优超参数 ==========")
print(xgb_grid.best_params_)
print(f"5折交叉验证最优AUC得分: {xgb_grid.best_score_:.4f}")

best_xgb = xgb_grid.best_estimator_
model_eval('XGBoost', best_xgb, X_train, X_test, y_train, y_test)  # 使用原始特征进行评估

# --- 8. 融合模型（Voting Ensemble） ---
print("\n8. 训练 Voting Ensemble 融合模型...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
    ],
    voting='soft'
)
# 使用原始训练集（树模型不需要缩放）
ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
print(f"\nEnsemble 测试集 AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")
print(f'\n===== Ensemble 测试集分类报告 =====')
print(classification_report(y_test, ensemble_pred))

# --- 9. 保存模型 ---
print("\n9. 正在保存模型...")
joblib.dump(best_rf, 'configs/rf_best_model.pkl')
joblib.dump(best_xgb, 'configs/xgb_best_model.pkl')
joblib.dump(ensemble, 'configs/ensemble_model.pkl')
joblib.dump(scaler, 'configs/scaler.pkl')
print("[OK] 已保存: rf_best_model.pkl, xgb_best_model.pkl, ensemble_model.pkl, scaler.pkl")

print("\n恭喜！模型训练阶段全部完成！")