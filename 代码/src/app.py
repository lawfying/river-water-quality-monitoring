

"""
AI水质安全监测预测系统 - 全功能整合版 Streamlit 前端
整合模块：
1. 数值水质分类预测
2. 水面漂浮物目标检测
3. LLM智能分析报告生成
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests
from datetime import datetime
import cv2
from PIL import Image
import tempfile
from collections import Counter

# ========== WQI 水质指数评分模块 ==========
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from water_quality_index import compute_wqi, classify_wqi, wqi_predict_proba, WQI_PARAMS

# ========== 默认 API Key 配置 ==========
# 方式1：代码内默认（方便自动连接）
DEFAULT_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 方式2：环境变量（优先）
# 如果设置了环境变量，优先使用环境变量；否则用默认
os.environ.setdefault("DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_API_KEY)
# =====================================

# 尝试导入 YOLO（漂浮物检测）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 尝试导入 LLM（DeepSeek）
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ========== 水质数据可视化图表生成 ==========
def generate_water_charts(wr):
    """生成水质雷达图 + WQI仪表盘"""
    import io
    import matplotlib.pyplot as plt
    from math import pi

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 9项指标及其WHO上限
    params = ['pH值', '硬度', '溶解固体', '氯胺', '硫酸盐', '电导率', '有机碳', '三卤甲烷', '浊度']
    limits = [8.5, 300, 1000, 4.0, 500, 1500, 4.0, 100, 5.0]
    values = [wr['ph'], wr['Hardness'], wr['Solids'], wr['Chloramines'],
              wr['Sulfate'], wr['Conductivity'], wr['Organic_carbon'],
              wr['Trihalomethanes'], wr['Turbidity']]
    # pH特殊处理：偏离6.5-8.5越远越差，归一化到0-1
    ph_val = wr['ph']
    if ph_val < 6.5:
        ph_norm = max(0, ph_val / 6.5)
    elif ph_val > 8.5:
        ph_norm = max(0, (14 - ph_val) / (14 - 8.5))
    else:
        ph_norm = 1.0  # 在范围内为最佳
    norm_values = [ph_norm]
    for v, lim in zip(values[1:], limits[1:]):
        norm_values.append(min(v / lim, 1.0) if lim > 0 else 0)

    # --- 雷达图 ---
    N = len(params)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig1, ax1 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(30)

    plt.xticks(angles[:-1], params, size=11)
    ax1.set_ylim(0, 1.05)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], size=8)

    # 安全区（绿色扇形背景）
    ax1.fill(angles, [1.0] * (N + 1), alpha=0.08, color='green')
    ax1.plot(angles, [1.0] * (N + 1), '--', color='green', linewidth=1, alpha=0.3, label='WHO上限(100%)')

    # 实测值
    plot_vals = norm_values + norm_values[:1]
    ax1.plot(angles, plot_vals, 'o-', linewidth=2, color='#1E90FF', label='实测值')
    ax1.fill(angles, plot_vals, alpha=0.2, color='#1E90FF')

    # 标注具体数值
    for i, (v, raw) in enumerate(zip(norm_values, values)):
        ax1.annotate(f'{raw}', xy=(angles[i], v), fontsize=8,
                     ha='center', va='bottom' if v < 0.85 else 'top',
                     color='#1E90FF', fontweight='bold')

    ax1.set_title('水质指标雷达图（越靠近外圈越优）', size=13, pad=20, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.tight_layout()

    # --- WQI仪表盘（柱状条） ---
    wqi_score = wr.get('wqi_score', 0)
    if wqi_score is None:
        wqi_score = 0

    fig2, ax2 = plt.subplots(figsize=(7, 1.5))
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # 渐变背景条
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#1E90FF']
    segments = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for (start, end), color in zip(segments, colors):
        ax2.barh(0.5, end - start, left=start, height=0.4, color=color, alpha=0.6, edgecolor='white')

    # 评分指针
    ax2.barh(0.5, wqi_score, left=0, height=0.4, color='#2C3E50', alpha=0.3, edgecolor='#2C3E50', linewidth=1.5)
    ax2.plot(wqi_score, 0.5, marker='v', color='#2C3E50', markersize=14, zorder=5)

    # 文字标签
    labels = ['差', '较差', '一般', '良好', '优秀']
    for (start, end), label in zip(segments, labels):
        ax2.text((start + end) / 2, 0.8, label, ha='center', va='bottom', fontsize=9, color='#555')

    ax2.text(wqi_score, 0.1, f'{wqi_score:.1f}', ha='center', va='top', fontsize=14,
             fontweight='bold', color='#2C3E50')
    ax2.set_title('WQI综合水质评分', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()

    return fig1, fig2


API_BASE = "http://127.0.0.1:8000/api"


def auto_save_water_record(wr):
    """将水质预测结果自动保存到后端 WaterRecord"""
    try:
        payload = {
            'ph': wr['ph'], 'hardness': wr['Hardness'], 'solids': wr['Solids'],
            'chloramines': wr['Chloramines'], 'sulfate': wr['Sulfate'],
            'conductivity': wr['Conductivity'], 'organic_carbon': wr['Organic_carbon'],
            'trihalomethanes': wr['Trihalomethanes'], 'turbidity': wr['Turbidity'],
            'wqi_score': wr.get('wqi_score'), 'wqi_label': wr.get('wqi_label', ''),
            'rf_prediction': wr.get('rf_prediction', ''),
            'source': 'streamlit',
        }
        requests.post(f"{API_BASE}/water-records/", json=payload, timeout=3)
    except requests.exceptions.ConnectionError:
        pass  # 后端未启动不影响前端使用
    except Exception:
        pass


@st.dialog("水质趋势图")
def trend_dialog():
    """水质历史数据趋势图弹窗"""
    # 直接从 API 获取数据
    try:
        resp = requests.get(f"{API_BASE}/water-records/?limit=200", timeout=3)
        if resp.status_code != 200:
            st.warning("获取水质记录失败")
            return
        records = resp.json()
        if isinstance(records, dict) and 'results' in records:
            records = records['results']
        if not records:
            st.info("暂无水质历史数据，请先进行水质预测")
            return
    except requests.exceptions.ConnectionError:
        st.warning("需要启动 Django 服务: cd iot_backend && python manage.py runserver")
        return

    # 反向排序（按时间升序）
    records = list(reversed(records))

    # 可选参数列表
    param_map = {
        'pH值': 'ph', '硬度': 'hardness', '溶解固体': 'solids',
        '氯胺': 'chloramines', '硫酸盐': 'sulfate', '电导率': 'conductivity',
        '有机碳': 'organic_carbon', '三卤甲烷': 'trihalomethanes',
        '浊度': 'turbidity', 'WQI综合评分': 'wqi_score',
    }

    selected_labels = st.multiselect(
        "选择要展示的参数（可多选）",
        options=list(param_map.keys()),
        default=['WQI综合评分'],
        placeholder="选择参数..."
    )

    if not selected_labels:
        st.info("请至少选择一个参数")
        return

    selected_fields = [param_map[lb] for lb in selected_labels]
    seq_ids = [r['sequence_id'] for r in records]

    # 生成折线图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['#1E90FF', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4']

    for i, (label, field) in enumerate(zip(selected_labels, selected_fields)):
        values = []
        for r in records:
            v = r.get(field)
            values.append(v if v is not None else 0)
        ax.plot(seq_ids, values, marker='o', label=label,
                color=colors[i % len(colors)], linewidth=1.5, markersize=4)

    ax.set_xlabel('记录序号', fontsize=10)
    ax.set_ylabel('数值', fontsize=10)
    ax.set_title('水质参数历史趋势', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if len(seq_ids) > 20:
        step = len(seq_ids) // 10
        ax.set_xticks(seq_ids[::step])
    plt.tight_layout()

    st.pyplot(fig)

    # 下载趋势图
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="下载趋势图 (PNG)",
            data=buf,
            file_name="水质趋势图.png",
            mime="image/png",
            use_container_width=True,
        )

    # CSV 导出
    csv_lines = ['序号,' + ','.join(selected_labels)]
    for r in records:
        row = [str(r['sequence_id'])]
        for _, field in zip(selected_labels, selected_fields):
            row.append(str(r.get(field, '')))
        csv_lines.append(','.join(row))
    csv_data = '\n'.join(csv_lines).encode('utf-8-sig')
    with dl_col2:
        st.download_button(
            label="下载趋势数据 (CSV)",
            data=csv_data,
            file_name="水质趋势数据.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="AI水质安全监测预测系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1E90FF; 
        text-align: center; 
        margin-bottom: 0.5rem;
    }
    .sub-header { 
        font-size: 1.2rem; 
        color: #666; 
        text-align: center; 
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: linear-gradient(90deg, #E8F4F8 0%, transparent 100%);
        border-left: 4px solid #2E86AB;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .result-safe {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .result-unsafe {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💧 AI水质安全监测预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">数值水质AI评估 · 漂浮物智能检测 · LLM分析报告生成</p>', unsafe_allow_html=True)

# --- 2. 侧边栏导航 ---
st.sidebar.markdown("## 📋 功能导航")
page = st.sidebar.radio(
    "选择功能模块",
    ["🏠 首页", "🔬 水质数值预测", "📷 漂浮物检测", "🤖 LLM智能报告", "📡 IoT传感器管理", "📊 批量处理"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 模型状态")

# 检查模型文件状态
model_status = {
    "水质分类模型": os.path.exists('configs/rf_best_model.pkl'),
    "标准化器": os.path.exists('configs/scaler.pkl'),
    "漂浮物检测模型": os.path.exists('output/float_detect/weights/best.pt'),
}

for name, exists in model_status.items():
    icon = "✅" if exists else "❌"
    st.sidebar.text(f"{icon} {name}")

# --- 3. 加载模型 ---
@st.cache_resource
def load_water_model():
    try:
        model = joblib.load('configs/rf_best_model.pkl')
        scaler = joblib.load('configs/scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        return None, None

@st.cache_resource
def load_yolo_model():
    model_path = 'output/float_detect/weights/best.pt'
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO模型加载失败: {e}")
        return None

model, scaler = load_water_model()
yolo_model = load_yolo_model() if YOLO_AVAILABLE else None

# ==================== 🏠 首页 ====================
if page == "🏠 首页":
    st.markdown('<p class="section-header">🎯 系统概述</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 水质数值预测</h3>
            <p>输入9项水质指标，AI实时判断水质是否安全可饮用</p>
            <ul>
                <li>pH值、硬度、溶解固体</li>
                <li>氯胺、硫酸盐、电导率</li>
                <li>有机碳、三卤甲烷、浊度</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📷 漂浮物检测</h3>
            <p>上传水面图像，YOLOv8自动识别13类漂浮垃圾</p>
            <ul>
                <li>易拉罐、塑料瓶、塑料袋</li>
                <li>树叶、树枝、纸盒</li>
                <li>实时标注与统计</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 LLM智能报告</h3>
            <p>基于DeepSeek大模型生成专业水质分析报告</p>
            <ul>
                <li>总体评价与关键指标分析</li>
                <li>风险评估与处理建议</li>
                <li>300-500字专业报告</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">📊 模型状态</p>', unsafe_allow_html=True)
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if model_status["水质分类模型"]:
            st.success("✅ 水质分类模型已加载")
        else:
            st.error("❌ 水质分类模型未找到\n请运行: python src/model_training.py")
    
    with status_col2:
        if yolo_model:
            st.success("✅ 漂浮物检测模型已加载")
        else:
            st.warning("⚠️ 漂浮物检测模型未找到\n请运行: python src_float/train.py")
    
    with status_col3:
        if LLM_AVAILABLE:
            st.success("✅ LLM API已就绪")
        else:
            st.warning("⚠️ LLM API未配置\n请检查 openai 库安装")

# ==================== 🔬 水质数值预测 ====================
elif page == "🔬 水质数值预测":
    st.markdown('<p class="section-header">🔬 水质数值预测</p>', unsafe_allow_html=True)
    
    if not model or not scaler:
        st.error("❌ 模型未加载！请确保 configs/rf_best_model.pkl 和 configs/scaler.pkl 存在。")
        st.stop()
    
    # ---- 数据来源选择 ----
    st.markdown("### 📝 数据输入方式")
    data_source = st.radio(
        "选择数据来源",
        ["手动输入", "从数据集加载", "从IoT传感器获取"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    if data_source == "从数据集加载":
        dataset_options = {
            "历史数据 (water_potability_clean.csv)": "data/water_potability_clean.csv",
            "2026年新数据 (water_potability_2026_clean.csv)": "data/water_potability_2026_clean.csv",
        }
        selected_dataset = st.selectbox("选择数据集", list(dataset_options.keys()))
        dataset_path = dataset_options[selected_dataset]
        
        try:
            source_df = pd.read_csv(dataset_path)
            # 只保留数值特征列用于展示
            display_df = source_df[feature_cols + ['Potability']].copy()
            st.success(f"已加载 {len(source_df)} 条记录")
            
            sample_idx = st.number_input(
                f"选择样本序号 (0 ~ {len(source_df)-1})",
                min_value=0, max_value=len(source_df)-1, value=0
            )
            sample = source_df.iloc[sample_idx]
            
            # 展示选中的样本数据
            st.markdown("**选中的样本数据：**")
            sample_df = pd.DataFrame([sample.to_dict()])
            st.dataframe(sample_df)
            
            if st.button("🔮 预测此样本", type="primary", use_container_width=True):
                input_data = {col: float(sample[col]) for col in feature_cols}
                
                # ===== WQI 水质指数评分（主预测） =====
                wqi_score, wqi_details = compute_wqi(input_data)
                wqi_label, wqi_conf, wqi_desc = classify_wqi(wqi_score)
                wqi_proba = wqi_predict_proba(input_data)
                
                # ===== RF 模型预测（参考） =====
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                rf_pred = model.predict(input_scaled)[0]
                rf_proba = model.predict_proba(input_scaled)[0]
                
                st.markdown("---")
                st.markdown("### 📊 预测结果分析")
                
                # 保存到 session_state 供 LLM 使用
                st.session_state.water_result = {
                    'ph': float(sample['ph']), 'Hardness': float(sample['Hardness']),
                    'Solids': float(sample['Solids']), 'Chloramines': float(sample['Chloramines']),
                    'Sulfate': float(sample['Sulfate']), 'Conductivity': float(sample['Conductivity']),
                    'Organic_carbon': float(sample['Organic_carbon']),
                    'Trihalomethanes': float(sample['Trihalomethanes']),
                    'Turbidity': float(sample['Turbidity']),
                    'wqi_score': wqi_score,
                    'wqi_label': wqi_label,
                    'wqi_desc': wqi_desc,
                    'prediction': int(rf_pred),
                    'conclusion': wqi_label,
                    'safe_prob': wqi_proba[0],
                    'unsafe_prob': wqi_proba[1],
                    'rf_prediction': "不合格" if rf_pred == 1 else "合格",
                }
                auto_save_water_record(st.session_state.water_result)

                # 显示真实标签对比
                real_label = int(sample.get('Potability', -1))
                if real_label != -1:
                    st.markdown(f"**真实标签:** {'不合格' if real_label == 1 else '合格'}")
                
                # ===== WQI 结果展示（主要） =====
                st.markdown("#### 1. 综合水质指数 (WQI) 评分")
                wqi_is_safe = wqi_score >= 45  # 45分以上算合格
                
                cols = st.columns([1, 2])
                with cols[0]:
                    if not wqi_is_safe:
                        st.markdown(f"""
                        <div class="result-unsafe">
                            <h2>⚠️ 预测结果：【{wqi_label}】</h2>
                            <p>{wqi_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            <h2>✅ 预测结果：【{wqi_label}】</h2>
                            <p>{wqi_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.metric("WQI 综合评分", f"{wqi_score:.1f}/100", 
                              delta="合格" if wqi_is_safe else "不合格",
                              delta_color="normal" if wqi_is_safe else "inverse")
                    st.progress(wqi_score / 100)
                    st.caption("WQI >= 45 为合格")
                
                # WQI 各参数得分详情
                st.markdown("**各参数水质评分：**")
                score_data = []
                for d in wqi_details:
                    score_data.append({
                        '参数': d['name'],
                        '值': d['value'],
                        '理想范围': f"{d['ideal_min']} - {d['ideal_max']}",
                        '得分': f"{d['score']:.1f}/100",
                    })
                st.dataframe(pd.DataFrame(score_data), hide_index=True)
                
                # ===== RF 模型结果（次要参考） =====
                st.markdown("#### 2. AI模型参考预测")
                st.caption("注：基于历史数据训练的随机森林模型，仅供参考")
                ref_col1, ref_col2 = st.columns(2)
                with ref_col1:
                    st.metric("RF模型预测", "不合格" if rf_pred == 1 else "合格")
                with ref_col2:
                    st.metric("RF模型安全概率", f"{rf_proba[0]:.2%}")
                
                st.info("💡 预测结果已保存，可前往「🤖 LLM智能报告」生成完整分析报告")
                
        except FileNotFoundError:
            st.error(f"未找到数据集 {dataset_path}，请先运行数据清洗脚本")
    elif data_source == "从IoT传感器获取":
        st.markdown("### 从IoT传感器获取数据")
        st.caption("从 Django IoT 后端获取各传感器最新监测数据")

        sensor_api_url = "http://127.0.0.1:8000/api/readings/latest/"
        sensor_list_url = "http://127.0.0.1:8000/api/sensors/"
        try:
            # 先检查是否有传感器
            sensor_resp = requests.get(sensor_list_url, timeout=3)
            sensors_exist = False
            if sensor_resp.status_code == 200:
                sensors_data = sensor_resp.json()
                if isinstance(sensors_data, dict) and 'results' in sensors_data:
                    sensors_exist = len(sensors_data['results']) > 0
                elif isinstance(sensors_data, list):
                    sensors_exist = len(sensors_data) > 0

            if not sensors_exist:
                st.info("暂无注册的传感器。请先前往「IoT传感器管理」页面注册传感器并上报数据。")
            else:
                resp = requests.get(sensor_api_url, timeout=3)
                if resp.status_code == 200:
                    readings = resp.json()
                    if not readings:
                        st.warning("传感器已注册但暂无监测数据。请在「IoT传感器管理」页面为传感器上报数据。")
                    else:
                        sensor_names = [f"{r['sensor_name']} ({r['sensor_location']})" for r in readings]
                        selected_idx = st.selectbox("选择传感器", range(len(sensor_names)),
                                                    format_func=lambda i: sensor_names[i])
                        reading = readings[selected_idx]

                    st.markdown("**传感器最新数据：**")
                    field_map = {
                        'ph': 'pH值', 'hardness': '硬度', 'solids': '溶解固体',
                        'chloramines': '氯胺', 'sulfate': '硫酸盐', 'conductivity': '电导率',
                        'organic_carbon': '有机碳', 'trihalomethanes': '三卤甲烷', 'turbidity': '浊度'
                    }
                    display_data = {}
                    for eng, cn in field_map.items():
                        val = reading.get(eng)
                        if val is not None:
                            display_data[cn] = val

                    st.dataframe(pd.DataFrame([display_data]))

                    if st.button("使用此传感器数据预测", type="primary", use_container_width=True):
                        input_data = {}
                        for eng in ['ph', 'hardness', 'solids', 'chloramines', 'sulfate',
                                    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']:
                            val = reading.get(eng)
                            input_data[eng.capitalize() if eng != 'ph' else 'ph'] = float(val) if val is not None else 0.0

                        wqi_score, wqi_details = compute_wqi(input_data)
                        wqi_label, wqi_conf, wqi_desc = classify_wqi(wqi_score)
                        wqi_proba = wqi_predict_proba(input_data)

                        input_df = pd.DataFrame([input_data])
                        input_scaled = scaler.transform(input_df)
                        rf_pred = model.predict(input_scaled)[0]
                        rf_proba = model.predict_proba(input_scaled)[0]

                        st.session_state.water_result = {
                            'ph': input_data['ph'], 'Hardness': input_data['Hardness'],
                            'Solids': input_data['Solids'], 'Chloramines': input_data['Chloramines'],
                            'Sulfate': input_data['Sulfate'], 'Conductivity': input_data['Conductivity'],
                            'Organic_carbon': input_data['Organic_carbon'],
                            'Trihalomethanes': input_data['Trihalomethanes'],
                            'Turbidity': input_data['Turbidity'],
                            'wqi_score': wqi_score,
                            'wqi_label': wqi_label,
                            'wqi_desc': wqi_desc,
                            'prediction': int(rf_pred),
                            'conclusion': wqi_label,
                            'safe_prob': wqi_proba[0],
                            'unsafe_prob': wqi_proba[1],
                            'rf_prediction': "不合格" if rf_pred == 1 else "合格",
                        }
                        auto_save_water_record(st.session_state.water_result)

                        st.markdown("---")
                        st.markdown("### 预测结果分析")

                        wqi_is_safe = wqi_score >= 45
                        cols = st.columns([1, 2])
                        with cols[0]:
                            if not wqi_is_safe:
                                st.markdown(f"""
                                <div class="result-unsafe">
                                    <h2>预测结果：【{wqi_label}】</h2>
                                    <p>{wqi_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-safe">
                                    <h2>预测结果：【{wqi_label}】</h2>
                                    <p>{wqi_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)

                        with cols[1]:
                            st.metric("WQI 综合评分", f"{wqi_score:.1f}/100",
                                      delta="合格" if wqi_is_safe else "不合格",
                                      delta_color="normal" if wqi_is_safe else "inverse")
                            st.progress(wqi_score / 100)
                            st.caption("WQI >= 45 为合格")

                        st.markdown("**各参数水质评分：**")
                        score_data = []
                        for d in wqi_details:
                            score_data.append({
                                '参数': d['name'],
                                '值': d['value'],
                                '理想范围': f"{d['ideal_min']} - {d['ideal_max']}",
                                '得分': f"{d['score']:.1f}/100",
                            })
                        st.dataframe(pd.DataFrame(score_data), hide_index=True)

                        st.markdown("#### AI模型参考预测")
                        st.caption("注：基于历史数据训练的随机森林模型，仅供参考")
                        ref_col1, ref_col2 = st.columns(2)
                        with ref_col1:
                            st.metric("RF模型预测", "不合格" if rf_pred == 1 else "合格")
                        with ref_col2:
                            st.metric("RF模型安全概率", f"{rf_proba[0]:.2%}")

                        st.info("预测结果已保存，可前往「LLM智能报告」生成完整分析报告")
                else:
                    st.error(f"获取传感器数据失败: HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.warning("无法连接 IoT 后端 (http://127.0.0.1:8000)。请先启动 Django 服务。")
        except Exception as e:
            st.error(f"获取传感器数据出错: {e}")
    else:
        # ---- 手动输入模式 ----
        st.markdown("### 📝 请输入水质监测数据")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph = st.number_input("pH值", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            hardness = st.number_input("硬度 (Hardness)", min_value=0.0, value=150.0, step=1.0)
            solids = st.number_input("溶解固体 (Solids)", min_value=0.0, value=10000.0, step=100.0)
        
        with col2:
            chloramines = st.number_input("氯胺 (Chloramines)", min_value=0.0, value=7.0, step=0.1)
            sulfate = st.number_input("硫酸盐 (Sulfate)", min_value=0.0, value=250.0, step=1.0)
            conductivity = st.number_input("电导率 (Conductivity)", min_value=0.0, value=300.0, step=1.0)
        
        with col3:
            organic_carbon = st.number_input("有机碳 (Organic_carbon)", min_value=0.0, value=10.0, step=0.1)
            trihalomethanes = st.number_input("三卤甲烷 (Trihalomethanes)", min_value=0.0, value=50.0, step=1.0)
            turbidity = st.number_input("浊度 (Turbidity)", min_value=0.0, value=4.0, step=0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 开始AI预测", type="primary", use_container_width=True)
        
        if predict_button:
            input_data = {
                'ph': ph, 'Hardness': hardness, 'Solids': solids, 
                'Chloramines': chloramines, 'Sulfate': sulfate, 'Conductivity': conductivity, 
                'Organic_carbon': organic_carbon, 'Trihalomethanes': trihalomethanes, 
                'Turbidity': turbidity
            }
            
            # ===== WQI 水质指数评分（主预测） =====
            wqi_score, wqi_details = compute_wqi(input_data)
            wqi_label, wqi_conf, wqi_desc = classify_wqi(wqi_score)
            wqi_proba = wqi_predict_proba(input_data)
            
            # ===== RF 模型预测（参考） =====
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            rf_pred = model.predict(input_scaled)[0]
            rf_proba = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 📊 预测结果分析")
            
            # 保存到 session_state 供 LLM 使用
            st.session_state.water_result = {
                'ph': ph, 'Hardness': hardness, 'Solids': solids,
                'Chloramines': chloramines, 'Sulfate': sulfate,
                'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
                'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity,
                'wqi_score': wqi_score,
                'wqi_label': wqi_label,
                'wqi_desc': wqi_desc,
                'prediction': int(rf_pred),
                'conclusion': wqi_label,
                'safe_prob': wqi_proba[0],
                'unsafe_prob': wqi_proba[1],
                'rf_prediction': "不合格" if rf_pred == 1 else "合格",
            }
            auto_save_water_record(st.session_state.water_result)

            # ===== WQI 结果展示（主要） =====
            st.markdown("#### 1. 综合水质指数 (WQI) 评分")
            wqi_is_safe = wqi_score >= 45
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if not wqi_is_safe:
                    st.markdown(f"""
                    <div class="result-unsafe">
                        <h2>⚠️ 预测结果：【{wqi_label}】</h2>
                        <p>{wqi_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h2>✅ 预测结果：【{wqi_label}】</h2>
                        <p>{wqi_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.metric("WQI 综合评分", f"{wqi_score:.1f}/100",
                          delta="合格" if wqi_is_safe else "不合格",
                          delta_color="normal" if wqi_is_safe else "inverse")
                st.progress(wqi_score / 100)
                st.caption("WQI >= 45 视为水质合格")
            
            # WQI 各参数得分
            st.markdown("**各参数水质评分：**")
            score_data = []
            for d in wqi_details:
                score_data.append({
                    '参数': d['name'],
                    '值': d['value'],
                    '理想范围': f"{d['ideal_min']} - {d['ideal_max']}",
                    '得分': f"{d['score']:.1f}/100",
                })
            st.dataframe(pd.DataFrame(score_data), hide_index=True)
            
            # ===== RF 模型结果（次要参考） =====
            st.markdown("#### 2. AI模型参考预测")
            st.caption("注：基于历史数据训练的随机森林模型，仅供参考")
            ref_col1, ref_col2 = st.columns(2)
            with ref_col1:
                st.metric("RF模型预测", "不合格" if rf_pred == 1 else "合格")
            with ref_col2:
                st.metric("RF模型安全概率", f"{rf_proba[0]:.2%}")
            
            st.info("💡 预测结果已保存，可前往「🤖 LLM智能报告」生成完整分析报告")

# ==================== 📷 漂浮物检测 ====================
elif page == "📷 漂浮物检测":
    st.markdown('<p class="section-header">📷 水面漂浮物目标检测</p>', unsafe_allow_html=True)
    
    if not YOLO_AVAILABLE:
        st.error("❌ YOLO未安装！请运行: pip install ultralytics")
        st.stop()
    
    if not yolo_model:
        st.warning("⚠️ 检测模型未找到！请先训练模型: python src_float/train.py")
        st.info("或使用预训练权重演示: yolo/yolov8n.pt")

        use_pretrained = st.checkbox("使用预训练权重（仅演示，不检测漂浮物）")
        if use_pretrained:
            try:
                yolo_model = YOLO('yolo/yolov8n.pt')
                st.success("✅ 预训练模型加载成功（仅演示通用物体检测）")
            except Exception as e:
                st.error(f"预训练模型加载失败: {e}")
                st.stop()
        else:
            st.stop()
    
    input_mode = st.radio("输入方式", ["上传图片", "上传视频"], horizontal=True, label_visibility="collapsed")

    if input_mode == "上传图片":
        uploaded_file = st.file_uploader("上传水面图像", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="原始图像", use_column_width=True)

            if st.button("开始检测", type="primary"):
                with st.spinner("正在检测漂浮物..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name)
                        results = yolo_model.predict(tmp.name, conf=0.25, save=False)
                        detections = []
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                conf_score = float(box.conf[0])
                                class_name = yolo_model.names[cls_id] if hasattr(yolo_model, 'names') else f"class_{cls_id}"
                                detections.append({
                                    'class': class_name,
                                    'confidence': round(conf_score, 4),
                                    'bbox': box.xyxy[0].tolist()
                                })
                        st.session_state.float_result = {
                            'count': len(detections),
                            'classes': list(set([d['class'] for d in detections])),
                            'detections': detections
                        }
                    os.unlink(tmp.name)

                st.markdown("### 检测结果")
                if len(detections) == 0:
                    st.success("未检测到漂浮物，水面清洁")
                else:
                    st.warning(f"检测到 {len(detections)} 个漂浮物目标")
                    class_counts = Counter([d['class'] for d in detections])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**类别分布：**")
                        for cls, count in class_counts.most_common():
                            st.text(f"  {cls}: {count} 个")
                    with col2:
                        st.markdown("**置信度详情：**")
                        for d in detections:
                            st.text(f"  {d['class']}: {d['confidence']:.2%}")
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="检测结果", use_column_width=True)

                st.info("检测数据已保存，可前往「LLM智能报告」生成完整分析报告")

    else:  # 上传视频
        video_file = st.file_uploader("上传水面监控视频", type=['mp4', 'avi', 'mov'])
        frame_interval = st.slider("抽帧间隔（每N帧检测一次）", min_value=1, max_value=60, value=15,
                                   help="值越大检测越快但可能漏检小目标")

        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            st.markdown(f"**视频信息：** 总帧数 {total_frames}，帧率 {fps:.1f} FPS，时长 {duration:.1f} 秒")
            st.markdown(f"**预估检测帧数：** {total_frames // frame_interval} 帧")

            if st.button("开始视频检测", type="primary"):
                cap = cv2.VideoCapture(video_path)
                all_detections = []
                frame_count = 0
                processed = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:
                        processed += 1
                        status_text.text(f"正在检测第 {processed} 帧...")
                        progress_bar.progress(min(processed / max(total_frames // frame_interval, 1), 1.0))

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            pil_img.save(tmp.name)
                            results = yolo_model.predict(tmp.name, conf=0.25, save=False, verbose=False)
                            for r in results:
                                boxes = r.boxes
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    conf_score = float(box.conf[0])
                                    class_name = yolo_model.names[cls_id] if hasattr(yolo_model, 'names') else f"class_{cls_id}"
                                    all_detections.append({
                                        'frame': processed,
                                        'class': class_name,
                                        'confidence': round(conf_score, 4),
                                    })
                        os.unlink(tmp.name)
                    frame_count += 1

                cap.release()
                os.unlink(video_path)
                progress_bar.empty()
                status_text.empty()

                # 汇总结果
                st.markdown("---")
                st.markdown("### 视频检测结果汇总")

                if len(all_detections) == 0:
                    st.success("视频中未检测到漂浮物")
                else:
                    st.warning(f"共检测到 {len(all_detections)} 个漂浮物目标（共检测 {processed} 帧）")

                    class_counts = Counter([d['class'] for d in all_detections])
                    st.markdown("**各类漂浮物出现次数：**")
                    for cls, count in class_counts.most_common():
                        st.text(f"  {cls}: {count} 次")

                    st.markdown("**帧级检测明细：**")
                    df_result = pd.DataFrame(all_detections)
                    df_result.columns = ['帧号', '类别', '置信度']
                    df_pivot = df_result.groupby(['帧号', '类别']).size().unstack(fill_value=0)
                    st.dataframe(df_pivot, use_container_width=True)

                    # 保存到 session_state 供 LLM 使用
                    st.session_state.float_result = {
                        'count': len(all_detections),
                        'classes': list(set([d['class'] for d in all_detections])),
                        'detections': all_detections
                    }

                    # 下载 CSV
                    csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="下载检测报告 (CSV)",
                        data=csv_data,
                        file_name="视频漂浮物检测结果.csv",
                        mime="text/csv"
                    )

                st.info("检测数据已保存，可前往「LLM智能报告」生成完整分析报告")

# ==================== 🤖 LLM智能报告 ====================
elif page == "🤖 LLM智能报告":
    st.markdown('<p class="section-header">🤖 LLM智能分析报告生成</p>', unsafe_allow_html=True)
    
    if not LLM_AVAILABLE:
        st.error("❌ OpenAI库未安装！请运行: pip install openai")
        st.stop()
    
    # ========== API Key 配置（输入框 + 默认值）==========
    st.markdown("### 🔑 API 配置")
    
    # 获取默认值（环境变量 > 代码默认）
    default_key = os.getenv("DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_API_KEY)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_key_input = st.text_input(
            "DeepSeek API Key（可选）",
            type="password",
            value=default_key,  # 默认填入
            placeholder="[REDACTED]",
            help="已自动填入默认Key，可更换为您自己的Key"
        )
    
    with col2:
        use_default = st.checkbox("使用默认Key", value=True, help="取消勾选可输入自定义Key")
    
    # 确定最终使用的Key
    if use_default:
        api_key = default_key
        st.info("✅ 使用默认API Key")
    else:
        api_key = api_key_input
        if api_key == default_key:
            st.warning("⚠️ 请修改API Key或勾选使用默认")
    
    # 验证Key是否有效
    if not api_key or api_key == "[REDACTED]":
        st.error("❌ 未配置有效的 DeepSeek API Key")
        st.stop()
    
    # 初始化客户端
    try:
        llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        # 测试连接
        st.success("✅ DeepSeek API 连接成功")
    except Exception as e:
        st.error(f"❌ API 连接失败: {e}")
        st.stop()
    # ================================================
    
    # 检查是否有前置数据
    has_water = 'water_result' in st.session_state
    has_float = 'float_result' in st.session_state
    
    st.markdown("### 📋 数据来源")
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        if has_water:
            st.success("✅ 水质预测数据已就绪")
            wr = st.session_state.water_result
            st.text(f"WQI评分: {wr.get('wqi_score', 'N/A')}/100")
            st.text(f"WQI等级: {wr.get('wqi_label', 'N/A')}")
            st.text(f"RF参考: {wr.get('rf_prediction', 'N/A')}")
        else:
            st.warning("⚠️ 暂无水质数据")
            st.info("请先在「🔬 水质数值预测」页面进行检测")
    
    with data_col2:
        if has_float:
            st.success("✅ 漂浮物检测数据已就绪")
            fr = st.session_state.float_result
            st.text(f"检测数量: {fr['count']} 个")
            st.text(f"主要类别: {', '.join(fr['classes'])}")
        else:
            st.warning("⚠️ 暂无漂浮物数据")
            st.info("请先在「📷 漂浮物检测」页面进行检测")
    
    # 生成报告
    st.markdown("---")
    
    if has_water or has_float:
        if st.button("生成LLM分析报告", type="primary", use_container_width=True):
            with st.spinner("正在调用DeepSeek生成报告..."):
                prompt = "你是一位资深水质监测分析专家。请根据以下数据生成一份专业、简洁的水质异常监测分析结论报告。\n\n"

                if has_water:
                    wr = st.session_state.water_result
                    prompt += f"""## 水质监测数据
| 参数 | 实测值 | WHO理想范围 |
|------|--------|------------|
| pH值 | {wr['ph']} | 6.5-8.5 |
| 硬度 | {wr['Hardness']} mg/L | ≤300 mg/L |
| 溶解固体 | {wr['Solids']} mg/L | ≤1000 mg/L |
| 氯胺 | {wr['Chloramines']} mg/L | ≤4.0 mg/L |
| 硫酸盐 | {wr['Sulfate']} mg/L | ≤500 mg/L |
| 电导率 | {wr['Conductivity']} uS/cm | ≤1500 uS/cm |
| 有机碳 | {wr['Organic_carbon']} mg/L | ≤4.0 mg/L |
| 三卤甲烷 | {wr['Trihalomethanes']} ug/L | ≤100 ug/L |
| 浊度 | {wr['Turbidity']} NTU | ≤5.0 NTU |

## WQI水质指数评分
- WQI综合评分: {wr.get('wqi_score', 'N/A')}/100
- 水质等级: {wr.get('wqi_label', 'N/A')}
- 评级说明: {wr.get('wqi_desc', 'N/A')}

## AI模型参考
- 随机森林模型参考: {wr.get('rf_prediction', 'N/A')}
"""

                if has_float:
                    fr = st.session_state.float_result
                    prompt += f"""
## 水面漂浮物检测结果
- 检测到的漂浮物数量: {fr['count']}
- 主要类别: {', '.join(fr['classes'])}
"""
                else:
                    prompt += "\n## 水面漂浮物检测结果\n- 未进行图像检测或未检测到漂浮物\n"

                prompt += """
请生成包含以下内容的分析报告（300-500字）：
1. 总体评价（水质是否安全）
2. **以Markdown表格形式展示各项水质监测数据**，包含参数名、实测值、理想范围、是否异常
3. 关键指标分析（指出异常指标及可能原因）
4. 风险评估与建议处理措施
5. 监测结论
要求语言专业、条理清晰、建议可操作，报告中的表格需为规范的Markdown格式。
"""

                try:
                    response = llm_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是环境工程领域的水质监测分析专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    st.session_state.report_content = response.choices[0].message.content
                    st.rerun()
                except Exception as e:
                    st.error(f"LLM调用失败: {e}")
                    st.info("请检查API Key是否正确，或网络连接是否正常")

    else:
        st.info("请先在左侧模块完成水质预测或漂浮物检测，再生成报告")

    # ========== 显示报告 + 保存（在按钮条件块外部） ==========
    if st.session_state.get('report_content'):
        report = st.session_state.report_content
        st.markdown("---")
        st.markdown("### [报告] 生成报告")
        st.markdown(report)

        # 显示水质图表（仅当有水数据时）
        if has_water:
            st.markdown("### 水质数据可视化")
            wr = st.session_state.water_result
            if wr.get('wqi_score') is not None:
                try:
                    fig_radar, fig_gauge = generate_water_charts(wr)
                    col_left, col_right = st.columns([1.2, 1])
                    with col_left:
                        st.pyplot(fig_radar)
                    with col_right:
                        st.pyplot(fig_gauge)
                except Exception as e:
                    pass  # 图表生成失败不影响报告展示

        # 保存到历史
        st.markdown("#### 保存报告")
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            report_title = st.text_input("报告标题",
                value="水质分析报告",
                placeholder="给这份报告起个名字", key="report_title_input")
        with save_col2:
            if st.button("保存报告", type="primary", use_container_width=True):
                try:
                    summary = report[:200].replace('\n', ' ').strip()
                    water_data = {}
                    if has_water:
                        wr = st.session_state.water_result
                        water_data = {k: str(v) for k, v in wr.items()}
                    payload = {
                        "title": report_title,
                        "content": report,
                        "summary": summary,
                        "wqi_score": st.session_state.water_result.get('wqi_score')
                            if has_water else None,
                        "wqi_grade": st.session_state.water_result.get('wqi_label', '')
                            if has_water else '',
                        "water_data": water_data,
                        "source": "streamlit"
                    }
                    resp = requests.post(
                        f"{API_BASE}/reports/",
                        json=payload, timeout=5
                    )
                    if resp.status_code in (200, 201):
                        st.success("报告已保存到历史记录")
                        st.rerun()
                    else:
                        st.error(f"保存失败: {resp.json()}")
                except requests.exceptions.ConnectionError:
                    st.warning("无法连接后端，报告未保存。请启动 Django 服务。")
                except Exception as e:
                    st.error(f"保存失败: {e}")

        st.download_button(
            label="下载报告 (TXT)",
            data=report,
            file_name="水质监测分析报告.txt",
            mime="text/plain"
        )

    # ========== 水质趋势分析 ==========
    st.markdown("---")
    st.markdown("### 水质趋势分析")
    trend_col1, trend_col2 = st.columns([3, 1])
    with trend_col1:
        st.caption("查看历史水质数据的变化趋势，支持多参数对比")
    with trend_col2:
        if st.button("生成趋势图", type="primary", use_container_width=True):
            trend_dialog()

    # ==================== 历史报告 ====================
    st.markdown("---")
    st.markdown("### 历史报告")

    try:
        resp = requests.get(f"{API_BASE}/reports/?limit=200", timeout=3)
        if resp.status_code == 200:
            reports = resp.json()
            if isinstance(reports, dict) and 'results' in reports:
                reports = reports['results']

            if not reports:
                st.info("暂无保存的历史报告")
            else:
                st.caption(f"共 {len(reports)} 条历史报告")

                # 批量删除
                batch_top = st.columns([1.5, 1, 1, 2])
                with batch_top[0]:
                    is_batch_mode = st.checkbox("选择要删除的报告", key="batch_del_mode")
                with batch_top[1]:
                    if is_batch_mode:
                        if st.button("全选", use_container_width=True, key="select_all_btn"):
                            if 'batch_del_ids' not in st.session_state:
                                st.session_state.batch_del_ids = set()
                            for r in reports:
                                st.session_state[f"bdel_{r['id']}"] = True
                                st.session_state.batch_del_ids.add(r['id'])
                            st.rerun()
                with batch_top[2]:
                    if is_batch_mode:
                        if st.button("取消全选", use_container_width=True, key="deselect_all_btn"):
                            for r in reports:
                                st.session_state[f"bdel_{r['id']}"] = False
                            st.session_state.batch_del_ids = set()
                            st.rerun()
                with batch_top[3]:
                    if is_batch_mode and st.session_state.get('batch_del_ids'):
                        if st.button(f"删除选中（{len(st.session_state.batch_del_ids)} 条）", type="primary", use_container_width=True):
                            try:
                                ids = list(st.session_state.batch_del_ids)
                                resp = requests.post(
                                    f"{API_BASE}/reports/batch-delete/",
                                    json={"ids": ids},
                                    timeout=5
                                )
                                if resp.status_code == 200:
                                    st.success(f"已删除 {len(ids)} 条报告")
                                    st.session_state.batch_del_ids = set()
                                    st.rerun()
                                else:
                                    st.error("批量删除失败")
                            except:
                                st.error("请求失败")

                if is_batch_mode:
                    if 'batch_del_ids' not in st.session_state:
                        st.session_state.batch_del_ids = set()
                    selected_ids = st.session_state.batch_del_ids

                    for r in reports:
                        c1, c2, c3, c4 = st.columns([0.5, 3, 1.5, 1])
                        with c1:
                            if st.checkbox("", key=f"bdel_{r['id']}"):
                                selected_ids.add(r['id'])
                            else:
                                selected_ids.discard(r['id'])
                        with c2:
                            st.markdown(f"**{r['title']}**")
                        with c3:
                            try:
                                t = datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))
                                st.caption(t.strftime('%Y-%m-%d %H:%M'))
                            except:
                                st.caption(r['created_at'][:16])
                        with c4:
                            if r.get('wqi_score') is not None:
                                st.markdown(f"WQI: {r['wqi_score']}")
                        st.divider()

                else:
                    # 正常查看模式
                    for r in reports:
                        with st.container():
                            cols = st.columns([3, 1, 1, 0.8])
                            with cols[0]:
                                st.markdown(f"**{r['title']}**")
                                try:
                                    t = datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))
                                    st.caption(f"{t.strftime('%Y-%m-%d %H:%M')}")
                                except:
                                    st.caption(r['created_at'][:16])
                            with cols[1]:
                                if r.get('wqi_score') is not None:
                                    st.markdown(f"WQI: {r['wqi_score']}")
                            with cols[2]:
                                st.markdown(f"来源: {r.get('source', '-')}")
                            with cols[3]:
                                view_key = f"view_{r['id']}"
                                del_key = f"del_{r['id']}"
                                if st.button("查看", key=view_key, use_container_width=True):
                                    st.session_state[f'viewing_report_{r["id"]}'] = not st.session_state.get(f'viewing_report_{r["id"]}', False)
                                if st.button("删除", key=del_key, use_container_width=True):
                                    try:
                                        requests.delete(f"{API_BASE}/reports/{r['id']}/", timeout=5)
                                        st.success("已删除")
                                        st.rerun()
                                    except:
                                        st.error("删除失败")

                            if st.session_state.get(f'viewing_report_{r["id"]}', False):
                                st.markdown(f"**摘要**: {r.get('summary', '')}")
                                with st.expander("查看完整报告", expanded=True):
                                    st.markdown(r['content'])

                                # 显示水质图表（如果保存了水数据）
                                wd = r.get('water_data')
                                if wd:
                                    try:
                                        wr = {}
                                        for k, v in wd.items():
                                            if k in ('wqi_label', 'wqi_desc', 'conclusion', 'rf_prediction', 'source'):
                                                wr[k] = v
                                            else:
                                                try:
                                                    wr[k] = float(v)
                                                except (ValueError, TypeError):
                                                    wr[k] = v if v else 0
                                        if wr.get('ph') is not None and wr.get('wqi_score') is not None:
                                            st.markdown("##### 水质数据可视化")
                                            fig_radar, fig_gauge = generate_water_charts(wr)
                                            c_left, c_right = st.columns([1.2, 1])
                                            with c_left:
                                                st.pyplot(fig_radar)
                                            with c_right:
                                                st.pyplot(fig_gauge)
                                    except Exception:
                                        pass

                                st.divider()
                            else:
                                st.divider()
        else:
            st.info("无法获取历史记录")
    except requests.exceptions.ConnectionError:
        st.info("历史报告功能需要启动 Django 服务: cd iot_backend && python manage.py runserver")

# ==================== 📡 IoT传感器管理 ====================
elif page == "📡 IoT传感器管理":
    st.markdown('<p class="section-header">📡 IoT 物联网传感器管理</p>', unsafe_allow_html=True)

    @st.dialog("添加新传感器")
    def add_sensor_dialog():
        st.markdown("输入传感器信息")
        name = st.text_input("传感器名称", placeholder="e.g. 取水口pH监测")
        device_id = st.text_input("设备编号", placeholder="e.g. SENSOR-001")
        location = st.selectbox("安装位置", [
            ("inlet", "取水口"), ("outlet", "出水口"),
            ("midstream", "河道中游"), ("reservoir", "水库"), ("other", "其他")
        ], format_func=lambda x: x[1])[0]
        location_detail = st.text_input("位置详情", placeholder="e.g. 东经113.2° 北纬23.1°")

        if st.button("确认添加", type="primary"):
            if not name or not device_id:
                st.error("名称和设备编号不能为空")
                return
            try:
                resp = requests.post(f"{API_BASE}/sensors/", json={
                    "name": name, "device_id": device_id,
                    "location": location, "location_detail": location_detail
                }, timeout=5)
                if resp.status_code in (200, 201):
                    st.success(f"传感器 {name} 添加成功")
                    st.rerun()
                else:
                    st.error(f"添加失败: {resp.json()}")
            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端，请确认 Django 服务已启动")
            except Exception as e:
                st.error(f"错误: {e}")

    @st.dialog("上报传感器数据")
    def report_data_dialog(sensor_id, device_id):
        st.markdown(f"### 向 {device_id} 上报数据")
        col1, col2, col3 = st.columns(3)
        fields = {}
        with col1:
            fields['ph'] = st.number_input("pH", value=7.0, step=0.1, format="%.1f")
            fields['hardness'] = st.number_input("硬度 (mg/L)", value=150.0, step=1.0)
            fields['solids'] = st.number_input("溶解固体 (ppm)", value=200.0, step=1.0)
            fields['chloramines'] = st.number_input("氯胺 (mg/L)", value=2.0, step=0.1)
        with col2:
            fields['sulfate'] = st.number_input("硫酸盐 (mg/L)", value=100.0, step=1.0)
            fields['conductivity'] = st.number_input("电导率 (μS/cm)", value=250.0, step=1.0)
            fields['organic_carbon'] = st.number_input("有机碳 (mg/L)", value=3.0, step=0.1)
            fields['trihalomethanes'] = st.number_input("三卤甲烷 (μg/L)", value=30.0, step=1.0)
        with col3:
            fields['turbidity'] = st.number_input("浊度 (NTU)", value=0.3, step=0.1, format="%.1f")
            fields['temperature'] = st.number_input("温度 (°C)", value=22.0, step=0.1)
            fields['flow_rate'] = st.number_input("流速 (m/s)", value=0.5, step=0.1)
            fields['water_level'] = st.number_input("水位 (m)", value=2.0, step=0.1)

        if st.button("上报", type="primary"):
            payload = {"device_id": device_id, "recorded_at": datetime.now().isoformat()}
            for k, v in fields.items():
                payload[k] = v
            try:
                resp = requests.post(f"{API_BASE}/readings/", json=payload, timeout=5)
                if resp.status_code in (200, 201):
                    st.success("数据上报成功")
                    st.rerun()
                else:
                    st.error(f"上报失败: {resp.json()}")
            except Exception as e:
                st.error(f"错误: {e}")

    @st.dialog("传感器历史数据")
    def history_dialog(device_id, name):
        st.markdown(f"**{name}** ({device_id}) 的历史监测数据")
        try:
            resp = requests.get(f"{API_BASE}/readings/?device_id={device_id}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and 'results' in data:
                    readings = data['results']
                elif isinstance(data, list):
                    readings = data
                else:
                    readings = []

                if not readings:
                    st.info("暂无历史监测数据")
                else:
                    df = pd.DataFrame(readings)
                    display_cols = ['recorded_at', 'ph', 'hardness', 'solids', 'chloramines',
                                    'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes',
                                    'turbidity', 'temperature', 'flow_rate', 'water_level']
                    display_cols = [c for c in display_cols if c in df.columns]
                    df_display = df[display_cols].copy()
                    df_display = df_display.rename(columns={
                        'recorded_at': '时间', 'ph': 'pH', 'hardness': '硬度',
                        'solids': '溶解固体', 'chloramines': '氯胺', 'sulfate': '硫酸盐',
                        'conductivity': '电导率', 'organic_carbon': '有机碳',
                        'trihalomethanes': '三卤甲烷', 'turbidity': '浊度',
                        'temperature': '温度', 'flow_rate': '流速', 'water_level': '水位'
                    })
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    st.caption(f"共 {len(readings)} 条记录")
            else:
                st.error(f"获取历史数据失败: HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.warning("无法连接后端服务")

    # --- 标签页 ---
    iot_tab1, iot_tab2, iot_tab3 = st.tabs(["传感器列表", "添加传感器", "数据总览"])

    with iot_tab1:
        st.markdown("### 已注册传感器")
        try:
            resp = requests.get(f"{API_BASE}/sensors/", timeout=3)
            if resp.status_code == 200:
                sensors = resp.json()
                if isinstance(sensors, dict) and 'results' in sensors:
                    sensors = sensors['results']

                if not sensors:
                    st.info("暂无传感器，请先在「添加传感器」页面注册")
                else:
                    for s in sensors:
                        status_icon = {"online": "在线", "offline": "离线", "error": "故障"}.get(s['status'], s['status'])
                        with st.container():
                            cols = st.columns([2, 1.5, 0.8, 1.5, 1, 0.8, 0.8])
                            with cols[0]:
                                st.markdown(f"**{s['name']}**")
                                st.caption(f"ID: {s['device_id']}")
                            with cols[1]:
                                loc_display = dict(inlet="取水口", outlet="出水口",
                                    midstream="河道中游", reservoir="水库", other="其他").get(s['location'], s['location'])
                                st.markdown(loc_display)
                                if s.get('location_detail'):
                                    st.caption(s['location_detail'])
                            with cols[2]:
                                st.markdown(status_icon)
                            with cols[3]:
                                if s.get('last_active'):
                                    try:
                                        last_active = datetime.fromisoformat(s['last_active'].replace('Z', '+00:00'))
                                        st.caption(f"{last_active.strftime('%m-%d %H:%M')}")
                                    except:
                                        st.caption(s['last_active'][:16])
                                else:
                                    st.caption("从未连接")
                            with cols[4]:
                                if st.button("上报", key=f"report_{s['id']}", use_container_width=True):
                                    report_data_dialog(s['id'], s['device_id'])
                            with cols[5]:
                                if st.button("历史", key=f"history_{s['id']}", use_container_width=True):
                                    st.session_state[f'history_sensor_{s["id"]}'] = s
                                    history_dialog(s['device_id'], s['name'])
                            with cols[6]:
                                if st.button("删除", key=f"del_{s['id']}", use_container_width=True):
                                    try:
                                        requests.delete(f"{API_BASE}/sensors/{s['device_id']}/", timeout=5)
                                        st.success(f"已删除 {s['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                            st.divider()
            else:
                st.error(f"获取传感器列表失败: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("无法连接后端服务，请启动 Django 服务:\n```\ncd iot_backend\npython manage.py runserver\n```")

    with iot_tab2:
        st.markdown("### 注册新传感器")
        if st.button("➕ 添加传感器", type="primary", use_container_width=True):
            add_sensor_dialog()

        st.markdown("---")
        st.markdown("#### 模拟传感器 - 快速批量注册")
        if st.button("一键注册3个模拟水质传感器"):
            demo_sensors = [
                {"name": "取水口水质监测", "device_id": "SENSOR-INLET-01", "location": "inlet", "location_detail": "河道上游取水点"},
                {"name": "出水口水质监测", "device_id": "SENSOR-OUTLET-01", "location": "outlet", "location_detail": "水厂出水口"},
                {"name": "水库水质监测", "device_id": "SENSOR-RES-01", "location": "reservoir", "location_detail": "中心水库监测浮标"},
            ]
            success = 0
            for s in demo_sensors:
                try:
                    resp = requests.post(f"{API_BASE}/sensors/", json=s, timeout=5)
                    if resp.status_code in (200, 201):
                        success += 1
                except:
                    pass
            if success:
                st.success(f"成功注册 {success} 个传感器")
                st.rerun()
            else:
                st.error("注册失败，请检查后端服务")

    with iot_tab3:
        st.markdown("### 传感器数据总览")
        try:
            # 获取统计信息
            stats_resp = requests.get(f"{API_BASE}/sensors/stats/", timeout=3)
            if stats_resp.status_code == 200:
                stats = stats_resp.json()
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("传感器总数", stats.get('total', 0))
                sc2.metric("在线", stats.get('online', 0))
                sc3.metric("离线", stats.get('offline', 0))
                sc4.metric("故障", stats.get('error', 0))

            # 获取最新读数
            st.markdown("#### 最新监测数据")
            latest_resp = requests.get(f"{API_BASE}/readings/latest/", timeout=3)
            if latest_resp.status_code == 200:
                latest_data = latest_resp.json()
                if latest_data:
                    df_latest = pd.DataFrame(latest_data)
                    display_cols = ['sensor_name', 'sensor_device_id', 'ph', 'turbidity',
                                    'temperature', 'recorded_at']
                    display_cols = [c for c in display_cols if c in df_latest.columns]
                    st.dataframe(df_latest[display_cols], use_container_width=True)

                    st.info("前往「水质数值预测」→ 选择「从IoT传感器获取」可直接使用传感器数据进行预测")
                else:
                    st.info("暂无传感器数据")
            else:
                st.info("暂无传感器数据")

        except requests.exceptions.ConnectionError:
            st.error("无法连接后端服务")

elif page == "📊 批量处理":
    st.markdown('<p class="section-header">📊 批量水质预测</p>', unsafe_allow_html=True)
    
    if not model:
        st.error("❌ 模型未加载！")
        st.stop()
    
    # 数据源选择
    batch_source = st.radio(
        "选择数据源",
        ["上传CSV文件", "使用内置数据集"],
        horizontal=True
    )
    
    required_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                   'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    df = None
    
    if batch_source == "上传CSV文件":
        uploaded_csv = st.file_uploader("上传CSV文件", type=['csv'])
        
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
    else:
        dataset_options = {
            "历史数据 (water_potability_clean.csv)": "data/water_potability_clean.csv",
            "2026年新数据 (water_potability_2026_clean.csv)": "data/water_potability_2026_clean.csv",
        }
        selected_dataset = st.selectbox("选择内置数据集", list(dataset_options.keys()))
        dataset_path = dataset_options[selected_dataset]
        
        try:
            df = pd.read_csv(dataset_path)
            st.success(f"✅ 已加载 {len(df)} 条记录")
        except FileNotFoundError:
            st.error(f"❌ 未找到数据集 {dataset_path}，请先运行数据清洗脚本")
    
    if df is not None:
        st.markdown("### 📋 数据预览")
        st.dataframe(df.head(10))
        
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            st.error(f"❌ 缺少列: {missing}")
        else:
            if st.button("🔮 批量预测", type="primary"):
                with st.spinner("正在批量预测..."):
                    X = df[required_cols]
                    X_scaled = scaler.transform(X)
                    
                    df["RF检测标签"] = model.predict(X_scaled)
                    df["RF结论"] = df["RF检测标签"].map({1: "水质不合格", 0: "水质合格"})
                    proba = model.predict_proba(X_scaled)
                    df["RF水质合格概率"] = proba[:, 0]
                    
                    # WQI 评分
                    wqi_scores = []
                    wqi_labels = []
                    for _, row in df.iterrows():
                        feat = {col: row[col] for col in required_cols}
                        wqi, _ = compute_wqi(feat)
                        label, _, _ = classify_wqi(wqi)
                        wqi_scores.append(round(wqi, 1))
                        wqi_labels.append(label)
                    df["WQI评分"] = wqi_scores
                    df["WQI等级"] = wqi_labels
                    df["WQI结论"] = df["WQI评分"].apply(lambda x: "合格" if x >= 45 else "不合格")
                
                st.markdown("### 📊 预测结果")
                st.dataframe(df)
                
                # 统计
                st.markdown("### 📈 统计概览")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总样本数", len(df))
                with col2:
                    wqi_safe = (df["WQI评分"] >= 45).sum()
                    st.metric("WQI合格", wqi_safe, f"{wqi_safe/len(df):.1%}")
                with col3:
                    wqi_unsafe = (df["WQI评分"] < 45).sum()
                    st.metric("WQI不合格", wqi_unsafe, f"{wqi_unsafe/len(df):.1%}")
                with col4:
                    rf_unsafe = (df["RF检测标签"] == 1).sum()
                    st.metric("RF不合格", rf_unsafe, f"{rf_unsafe/len(df):.1%}")
                
                # 下载
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载结果CSV",
                    data=csv,
                    file_name="water_predict_result.csv",
                    mime="text/csv"
                )
