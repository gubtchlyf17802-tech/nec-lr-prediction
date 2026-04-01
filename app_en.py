# ============================================================================
# NEC Early Surgical Risk Prediction Tool - Streamlit Web App (English)
# ============================================================================

import streamlit as st
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEC Surgical Risk Prediction",
    page_icon="🏥",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
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
    st.error(f"Model loading failed: {e}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 NEC Early Surgical Risk Prediction")
st.markdown(
    "A **Logistic Regression nomogram model** for predicting the risk of surgical "
    "intervention within **72 hours** of NEC diagnosis."
)
st.caption("Training AUC = 0.887 (n=356) | Temporal validation AUC = 0.816 (n=126)")
st.divider()

# ── Input section ─────────────────────────────────────────────────────────────
st.subheader("📋 Patient Clinical Information")
st.markdown("*All laboratory values should be from within 24 hours of NEC diagnosis*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Imaging & Clinical Features**")

    bw_input = st.radio(
        "Birth Weight",
        options=["≥ 2500 g (Normal)", "< 2500 g (Low Birth Weight)"],
        horizontal=False
    )
    bw_catLBW = 1 if "Low Birth Weight" in bw_input else 0

    xray_input = st.radio(
        "X-ray Fixed Bowel Loops",
        options=["Absent", "Present"],
        horizontal=True
    )
    xray_fixed_loops = 1 if xray_input == "Present" else 0

    us_input = st.radio(
        "Ultrasound Complex Ascites",
        options=["Absent", "Present"],
        horizontal=True
    )
    us_complex_ascites = 1 if us_input == "Present" else 0

with col2:
    st.markdown("**Laboratory Parameters**")

    fibrinogen = st.number_input(
        "Fibrinogen (g/L)",
        min_value=0.0, max_value=20.0,
        value=float(round(medians.get('fibrinogen_gL_24h', 2.5), 1)),
        step=0.1, format="%.1f"
    )
    crp = st.number_input(
        "C-Reactive Protein (mg/L)",
        min_value=0.0, max_value=500.0,
        value=float(round(medians.get('crp_mgL_24h', 20.0), 0)),
        step=1.0, format="%.0f"
    )
    neut_percent = st.number_input(
        "Neutrophil Percentage (%)",
        min_value=0.0, max_value=100.0,
        value=float(round(medians.get('neut_percent_24h', 60.0), 1)),
        step=0.5, format="%.1f"
    )
    glucose = st.number_input(
        "Glucose (mmol/L)",
        min_value=0.0, max_value=30.0,
        value=float(round(medians.get('glucose_mmolL_24h', 5.0), 1)),
        step=0.1, format="%.1f"
    )
    na = st.number_input(
        "Sodium (mmol/L)",
        min_value=100.0, max_value=180.0,
        value=float(round(medians.get('na_24h', 138.0), 1)),
        step=0.5, format="%.1f"
    )
    albumin = st.number_input(
        "Albumin (g/L)",
        min_value=0.0, max_value=60.0,
        value=float(round(medians.get('albumin_24h', 30.0), 1)),
        step=0.5, format="%.1f"
    )

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button(
    "🔍 Calculate Surgical Risk",
    use_container_width=True,
    type="primary",
    disabled=not model_loaded
)

if predict_btn and model_loaded:

    feature_values = np.array([[
        xray_fixed_loops,
        fibrinogen,
        bw_catLBW,
        glucose,
        na,
        albumin,
        neut_percent,
        us_complex_ascites,
        crp
    ]])

    feature_scaled = scaler.transform(feature_values)
    prob           = model.predict_proba(feature_scaled)[0][1]
    prob_pct       = prob * 100

    # Risk stratification
    if prob < 0.30:
        risk_level = "Low Risk"
        risk_color = "🟢"
        advice     = "Predicted risk is low. Continue conservative management with regular monitoring every 6–8 hours."
    elif prob < 0.60:
        risk_level = "Intermediate Risk"
        risk_color = "🟡"
        advice     = "Predicted risk is moderate. Surgical consultation is recommended with increased monitoring frequency."
    else:
        risk_level = "High Risk"
        risk_color = "🔴"
        advice     = "Predicted risk is high. Prompt surgical evaluation is recommended with preparation for operative intervention."

    # Results display
    st.subheader("📊 Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Surgical Risk Probability", f"{prob_pct:.1f}%")
    m2.metric("Risk Level", f"{risk_color} {risk_level}")
    m3.metric("Model Reference AUC", "0.816")

    st.progress(float(min(prob, 1.0)))
    st.info(f"**Clinical Recommendation:** {advice}")

    # Input summary
    with st.expander("View Input Summary"):
        summary = {
            "Low Birth Weight":           "Yes" if bw_catLBW else "No",
            "X-ray Fixed Bowel Loops":    "Present" if xray_fixed_loops else "Absent",
            "Ultrasound Complex Ascites": "Present" if us_complex_ascites else "Absent",
            "Fibrinogen (g/L)":           fibrinogen,
            "CRP (mg/L)":                 crp,
            "Neutrophil % (%)":           neut_percent,
            "Glucose (mmol/L)":           glucose,
            "Sodium (mmol/L)":            na,
            "Albumin (g/L)":              albumin,
        }
        for k, v in summary.items():
            st.write(f"- **{k}**: {v}")

    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: This tool is intended for clinical decision support only and does not "
        "replace the clinical judgment of the treating physician. The model was developed from "
        "single-centre retrospective data; external validation is recommended before use in other settings."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Model Information")
    st.markdown("""
| Item | Detail |
|------|--------|
| Outcome | Surgery within 72h of NEC diagnosis |
| Method | Logistic Regression Nomogram |
| Training set | n=356 (2022–2024) |
| Validation set | n=126 (2025, temporal) |
| Training AUC | 0.887 |
| Validation AUC | 0.816 |
| Validation Brier | 0.163 |
    """)

    st.divider()
    st.markdown("### 📋 9 Predictors")
    st.markdown("""
**Imaging**
- X-ray Fixed Bowel Loops
- Ultrasound Complex Ascites

**Clinical**
- Low Birth Weight

**Inflammatory**
- C-Reactive Protein
- Neutrophil Percentage

**Coagulation / Protein**
- Fibrinogen
- Albumin

**Metabolic**
- Glucose
- Sodium
    """)

    st.divider()
    st.markdown("### ⚠️ Intended Population")
    st.caption(
        "This tool is intended for NEC patients without definitive absolute surgical indications "
        "(e.g., pneumoperitoneum). Patients with clear surgical indications should proceed "
        "directly to surgical management without delay."
    )
