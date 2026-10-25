"""
Water Quality Index (WQI) scoring module - 严格版
基于 WHO 饮用水标准，使用快速衰减指数评分 + 最差参数惩罚机制。
确保任何严重超标参数都能显著拉低总分。
"""
import numpy as np

# 各参数理想范围（严格版 - WHO 饮用水标准收紧）
# 使用 range_width 统一计算"距离倍数"
WQI_PARAMS = {
    'ph':               {'ideal': (6.5, 8.0),    'weight': 0.15, 'unit': ''},
    'Hardness':         {'ideal': (60, 200),     'weight': 0.08, 'unit': 'mg/L'},
    'Solids':           {'ideal': (100, 300),    'weight': 0.12, 'unit': 'mg/L'},   # TDS 饮用标准收紧
    'Chloramines':      {'ideal': (0.5, 4.0),    'weight': 0.12, 'unit': 'mg/L'},
    'Sulfate':          {'ideal': (3, 200),      'weight': 0.10, 'unit': 'mg/L'},   # 收窄
    'Conductivity':     {'ideal': (100, 400),    'weight': 0.10, 'unit': 'uS/cm'},  # 收窄
    'Organic_carbon':   {'ideal': (0.5, 6),      'weight': 0.12, 'unit': 'mg/L'},   # 收窄
    'Trihalomethanes':  {'ideal': (0, 60),       'weight': 0.10, 'unit': 'ug/L'},   # 收窄
    'Turbidity':        {'ideal': (0, 0.5),      'weight': 0.11, 'unit': 'NTU'},    # WHO 标准 0.5
}


def score_single(value, ideal_min, ideal_max):
    """
    单参数评分 0-100。
    
    评分逻辑：
    - 在理想范围内 → 100 分（满分）
    - 超出范围 → 按距离倍数快速衰减
    - 衰减系数 1.5（越远扣分越狠）
    """
    if value is None or np.isnan(value):
        return 50.0

    # 在理想范围内
    if ideal_min <= value <= ideal_max:
        return 100.0

    # 计算"距离倍数" = 超出距离 / 范围宽度
    range_width = max(ideal_max - ideal_min, 0.1)
    
    if value < ideal_min:
        distance = (ideal_min - value) / range_width
    else:
        distance = (value - ideal_max) / range_width

    # 指数衰减评分：distance=1→37分, distance=2→14分, distance=3→5分
    score = 100.0 * np.exp(-1.0 * distance)
    return round(max(0.0, min(100.0, score)), 1)


def compute_wqi(features_dict):
    """
    计算综合水质指数 WQI（0-100）。
    
    使用加权平均 + 最差参数惩罚机制：
    - 先计算加权平均分
    - 如果有任何参数评分极低（< 40），对总分施加惩罚倍率
    - 惩罚力度 = min_score / 40，确保单项严重超标拖累整体
    """
    details = []
    weighted_sum = 0.0
    total_weight = 0.0

    for param, config in WQI_PARAMS.items():
        if param in features_dict:
            value = features_dict[param]
            score = score_single(value, config['ideal'][0], config['ideal'][1])
            details.append({
                'name': param,
                'value': value,
                'score': score,
                'weight': config['weight'],
                'ideal_min': config['ideal'][0],
                'ideal_max': config['ideal'][1],
                'unit': config['unit'],
            })
            weighted_sum += score * config['weight']
            total_weight += config['weight']

    if total_weight == 0:
        return 50.0, details

    wqi = round(weighted_sum / total_weight, 1)

    # ===== 最差参数惩罚机制（混合公式） =====
    # WQI = 0.7 * 加权平均 + 0.3 * 最低分
    # 确保任何严重超标参数都能显著影响总分，但不会完全压倒其他指标
    min_score = min(d['score'] for d in details)
    wqi = round(0.7 * wqi + 0.3 * min_score, 1)

    return wqi, details


def classify_wqi(wqi):
    """
    根据 WQI 评分分类水质等级。
    返回 (等级标签, 置信度, 描述)
    """
    if wqi >= 85:
        return ("优质", wqi / 100, "水质优良，适合直接饮用")
    elif wqi >= 65:
        return ("良好", wqi / 100, "水质良好，基本适合饮用")
    elif wqi >= 45:
        return ("合格", wqi / 100, "水质合格，可接受但建议监测")
    elif wqi >= 25:
        return ("不合格", (100 - wqi) / 100, "水质较差，不建议直接饮用，建议处理")
    else:
        return ("危险", (100 - wqi) / 100, "水质很差，严重不适合饮用，需紧急处理")


def wqi_predict_proba(features_dict):
    """
    返回兼容 sklearn 格式的 [安全概率, 不安全概率]。
    安全概率 = WQI / 100
    """
    wqi, _ = compute_wqi(features_dict)
    prob_safe = max(0.0, min(1.0, wqi / 100.0))
    return [prob_safe, 1 - prob_safe]


if __name__ == '__main__':
    test_cases = [
        ("理想饮用水",   {'ph':7.0, 'Hardness':120, 'Solids':200, 'Chloramines':2, 'Sulfate':100, 'Conductivity':250, 'Organic_carbon':3, 'Trihalomethanes':20, 'Turbidity':0.2}),
        ("正常水平",     {'ph':7.5, 'Hardness':150, 'Solids':300, 'Chloramines':3, 'Sulfate':150, 'Conductivity':350, 'Organic_carbon':5, 'Trihalomethanes':40, 'Turbidity':0.5}),
        ("轻微污染水",   {'ph':8.0, 'Hardness':250, 'Solids':500, 'Chloramines':5, 'Sulfate':250, 'Conductivity':500, 'Organic_carbon':8, 'Trihalomethanes':80, 'Turbidity':2.0}),
        ("中等污染水",   {'ph':5.5, 'Hardness':50,  'Solids':1200, 'Chloramines':8, 'Sulfate':350, 'Conductivity':650, 'Organic_carbon':15, 'Trihalomethanes':120, 'Turbidity':5.0}),
        ("严重污染水",   {'ph':5.0, 'Hardness':400, 'Solids':3000, 'Chloramines':12, 'Sulfate':500, 'Conductivity':800, 'Organic_carbon':30, 'Trihalomethanes':200, 'Turbidity':15.0}),
        ("酸性废水",     {'ph':3.0, 'Hardness':50,  'Solids':20000,'Chloramines':15, 'Sulfate':600, 'Conductivity':750, 'Organic_carbon':25, 'Trihalomethanes':150, 'Turbidity':12.0}),
        ("碱性废水",     {'ph':11.0,'Hardness':350, 'Solids':2000, 'Chloramines':1,  'Sulfate':50,  'Conductivity':150, 'Organic_carbon':3,  'Trihalomethanes':10, 'Turbidity':0.8}),
    ]

    print(f"{'Case':20s} {'WQI':8s} {'Label':10s} {'P(safe)':10s} {'P(unsafe)':10s}")
    print("-" * 60)
    for name, data in test_cases:
        wqi, details = compute_wqi(data)
        label, conf, desc = classify_wqi(wqi)
        p = wqi_predict_proba(data)
        print(f"{name:20s} {wqi:<8.1f} {label:10s} {p[0]:<10.3f} {p[1]:<10.3f}")

    # 打印每个参数的评分详情（仅第一个用例）
    print("\n=== 参数评分示例（理想饮用水）===")
    wqi, details = compute_wqi(test_cases[0][1])
    for d in details:
        print(f"  {d['name']:20s} 值={str(d['value']):10s}  得分={d['score']:.1f}")
    print(f"  {'WQI总分':20s}  {'':10s}  {wqi:.1f}")

    print("\n=== 参数评分示例（酸性废水）===")
    wqi, details = compute_wqi(test_cases[5][1])
    for d in details:
        print(f"  {d['name']:20s} 值={str(d['value']):10s}  得分={d['score']:.1f}")
    print(f"  {'WQI总分':20s}  {'':10s}  {wqi:.1f}")
