# ============================================================================
# NEC早期手术风险预测工具 - Streamlit Web App
# ============================================================================

import streamlit as st
import numpy as np
import pickle

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEC手术风险预测",
    page_icon="🏥",
    layout="centered"
)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('lr_model.pkl',  'rb') as f: model   = pickle.load(f)
    with open('scaler.pkl',    'rb') as f: scaler  = pickle.load(f)
    with open('medians.pkl',   'rb') as f: medians = pickle.load(f)
    return model, scaler, medians

try:
    model, scaler, medians = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"模型文件加载失败：{e}")

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.title("🏥 NEC 早期手术风险预测")
st.markdown(
    "基于多因素 Logistic 回归列线图模型，预测 NEC 患儿确诊后 **72小时内** 手术干预风险。"
)
st.caption("训练集 AUC=0.887（n=356），时间外验证集 AUC=0.816（n=126）")
st.divider()

# ── 输入区域 ──────────────────────────────────────────────────────────────────
st.subheader("📋 输入患儿临床信息")
st.markdown("*所有实验室指标均取 NEC 确诊后 24 小时内最近时间点数据*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**影像学与基础信息**")

    bw_input = st.radio(
        "出生体重",
        options=["≥ 2500 g（正常）", "< 2500 g（低出生体重）"],
        horizontal=False
    )
    bw_catLBW = 1 if "低出生体重" in bw_input else 0

    xray_input = st.radio(
        "X线固定肠袢",
        options=["无", "有"],
        horizontal=True
    )
    xray_fixed_loops = 1 if xray_input == "有" else 0

    us_input = st.radio(
        "超声复杂腹水",
        options=["无", "有"],
        horizontal=True
    )
    us_complex_ascites = 1 if us_input == "有" else 0

with col2:
    st.markdown("**实验室指标**")

    fibrinogen = st.number_input(
        "纤维蛋白原（g/L）",
        min_value=0.0, max_value=20.0,
        value=float(round(medians.get('fibrinogen_gL_24h', 2.5), 1)),
        step=0.1, format="%.1f"
    )
    crp = st.number_input(
        "C反应蛋白（mg/L）",
        min_value=0.0, max_value=500.0,
        value=float(round(medians.get('crp_mgL_24h', 20.0), 0)),
        step=1.0, format="%.0f"
    )
    neut_percent = st.number_input(
        "中性粒细胞百分比（%）",
        min_value=0.0, max_value=100.0,
        value=float(round(medians.get('neut_percent_24h', 60.0), 1)),
        step=0.5, format="%.1f"
    )
    glucose = st.number_input(
        "血糖（mmol/L）",
        min_value=0.0, max_value=30.0,
        value=float(round(medians.get('glucose_mmolL_24h', 5.0), 1)),
        step=0.1, format="%.1f"
    )
    na = st.number_input(
        "血钠（mmol/L）",
        min_value=100.0, max_value=180.0,
        value=float(round(medians.get('na_24h', 138.0), 1)),
        step=0.5, format="%.1f"
    )
    albumin = st.number_input(
        "白蛋白（g/L）",
        min_value=0.0, max_value=60.0,
        value=float(round(medians.get('albumin_24h', 30.0), 1)),
        step=0.5, format="%.1f"
    )

# ── 预测按钮 ──────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button(
    "🔍 计算手术风险",
    use_container_width=True,
    type="primary",
    disabled=not model_loaded
)

if predict_btn and model_loaded:

    # 特征顺序必须与训练时完全一致
    feature_values = np.array([[
        xray_fixed_loops,    # xray_fixed_loops
        fibrinogen,          # fibrinogen_gL_24h
        bw_catLBW,           # bw_catLBW
        glucose,             # glucose_mmolL_24h
        na,                  # na_24h
        albumin,             # albumin_24h
        neut_percent,        # neut_percent_24h
        us_complex_ascites,  # us_complex_ascites
        crp                  # crp_mgL_24h
    ]])

    feature_scaled = scaler.transform(feature_values)
    prob           = model.predict_proba(feature_scaled)[0][1]
    prob_pct       = prob * 100

    # 风险分层
    if prob < 0.30:
        risk_level = "低风险"
        risk_color = "🟢"
        bar_color  = "normal"
        advice     = "当前预测风险较低，建议继续保守治疗，每6–8小时评估一次病情变化。"
    elif prob < 0.60:
        risk_level = "中等风险"
        risk_color = "🟡"
        bar_color  = "off"
        advice     = "当前预测风险处于中等水平，建议外科会诊评估，加强生命体征及实验室监测频率。"
    else:
        risk_level = "高风险"
        risk_color = "🔴"
        bar_color  = "inverse"
        advice     = "当前预测风险较高，建议及时外科评估，充分准备手术干预。"

    # 结果展示
    st.subheader("📊 预测结果")

    m1, m2, m3 = st.columns(3)
    m1.metric("手术风险概率", f"{prob_pct:.1f}%")
    m2.metric("风险等级", f"{risk_color} {risk_level}")
    m3.metric("模型置信度参考", "AUC 0.816")

    st.progress(float(min(prob, 1.0)))

    st.info(f"**临床建议：** {advice}")

    # 输入值汇总
    with st.expander("查看本次输入汇总"):
        summary = {
            "低出生体重":      "是" if bw_catLBW else "否",
            "X线固定肠袢":     "有" if xray_fixed_loops else "无",
            "超声复杂腹水":    "有" if us_complex_ascites else "无",
            "纤维蛋白原(g/L)": fibrinogen,
            "CRP(mg/L)":       crp,
            "中性粒细胞%":     neut_percent,
            "血糖(mmol/L)":    glucose,
            "血钠(mmol/L)":    na,
            "白蛋白(g/L)":     albumin,
        }
        for k, v in summary.items():
            st.write(f"- **{k}**：{v}")

    st.divider()
    st.caption(
        "⚠️ **免责声明**：本工具仅供临床辅助参考，不能替代临床医生的综合判断与决策。"
        "模型基于单中心回顾性数据开发，在不同中心应用前建议进行本地验证。"
    )

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 模型信息")
    st.markdown("""
| 项目 | 内容 |
|------|------|
| 预测目标 | 72h内手术干预 |
| 建模方法 | Logistic回归 |
| 训练集 | n=356（2022–2024） |
| 验证集 | n=126（2025，时间外） |
| 训练AUC | 0.887 |
| 验证AUC | 0.816 |
| 验证Brier | 0.163 |
    """)

    st.divider()
    st.markdown("### 📋 9个预测变量")
    st.markdown("""
**影像学**
- X线固定肠袢
- 超声复杂腹水

**基础信息**
- 低出生体重

**炎症指标**
- CRP
- 中性粒细胞百分比

**凝血-蛋白**
- 纤维蛋白原
- 白蛋白

**代谢指标**
- 血糖
- 血钠
    """)

    st.divider()
    st.markdown("### ⚠️ 适用人群")
    st.caption(
        "本工具适用于尚未出现明确绝对手术指征（如气腹）的NEC患儿。"
        "对于已具备明确手术指征的病例，应即时外科处理，无需使用本工具评估。"
    )
