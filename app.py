import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.tree import plot_tree, export_text, _tree
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark clinical aesthetic
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1525;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #8ab4d4 !important;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Headers */
h1 { font-family: 'DM Serif Display', serif !important; color: #ffffff !important; }
h2 { font-family: 'DM Serif Display', serif !important; color: #c8d8f0 !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #8ab4d4 !important; font-weight: 500 !important; }

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #e74c3c;
}
.card-safe {
    border-left: 3px solid #2ecc71;
}
.card-info {
    border-left: 3px solid #3498db;
}

/* Result banner */
.result-danger {
    background: linear-gradient(135deg, #2d0a0a 0%, #1a0f0f 100%);
    border: 1px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(231,76,60,0.15);
}
.result-safe {
    background: linear-gradient(135deg, #0a2d14 0%, #0f1a10 100%);
    border: 1px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(46,204,113,0.12);
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0.3rem 0;
}
.result-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    opacity: 0.7;
    text-transform: uppercase;
}

/* Metric boxes */
.metric-box {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8ab4d4;
    margin-top: 0.3rem;
}

/* Rule path */
.rule-step {
    background: #0f1525;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #a8c8e8;
}
.rule-step-left  { border-left: 3px solid #3498db; }
.rule-step-right { border-left: 3px solid #f39c12; }
.rule-leaf-danger { border-left: 3px solid #e74c3c; color: #ff8080; }
.rule-leaf-safe   { border-left: 3px solid #2ecc71; color: #80ff9f; }

/* Prob bar */
.prob-container { margin: 1rem 0; }
.prob-label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #8ab4d4; margin-bottom: 0.3rem; }
.prob-bar-bg { background: #1e2d4a; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; }
.prob-bar-fill { height: 10px; border-radius: 6px; transition: width 0.8s ease; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 1.5rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff6b5b, #e74c3c) !important;
    box-shadow: 0 4px 20px rgba(231,76,60,0.4) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #e74c3c !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8ab4d4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #e74c3c !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌  `heart_disease_model.pkl` not found.  Place it in the same folder as `app.py`.")
    st.stop()

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']
FEATURE_LABELS = ['Age','Sex','Chest Pain Type','Resting BP','Cholesterol',
                  'Fasting Blood Sugar','Rest ECG','Max Heart Rate',
                  'Exercise Angina','ST Depression','ST Slope',
                  'Major Vessels','Thalassemia']
CLASS_NAMES = ['No Disease', 'Heart Disease']


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def get_decision_path(model, input_array):
    tree_    = model.tree_
    feat_idx = model.tree_.feature
    node     = 0
    path     = []
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        fidx      = feat_idx[node]
        fname     = FEATURE_LABELS[fidx]
        thr       = tree_.threshold[node]
        val       = input_array[0, fidx]
        if val <= thr:
            direction = "LEFT  ≤"
            node = tree_.children_left[node]
            side = "left"
        else:
            direction = "RIGHT >"
            node = tree_.children_right[node]
            side = "right"
        path.append({"feature": fname, "value": val, "threshold": thr,
                     "direction": direction, "side": side})
    vals     = tree_.value[node][0]
    cls_idx  = int(np.argmax(vals))
    conf     = vals[cls_idx] / vals.sum() * 100
    path.append({"leaf": True, "class": CLASS_NAMES[cls_idx], "confidence": conf,
                 "samples": int(vals.sum()), "cls_idx": cls_idx})
    return path


def risk_color(prob):
    if prob < 0.30: return "#2ecc71"
    if prob < 0.55: return "#f39c12"
    return "#e74c3c"


def risk_label(prob):
    if prob < 0.30: return "LOW RISK"
    if prob < 0.55: return "MODERATE RISK"
    return "HIGH RISK"


# ─────────────────────────────────────────────────────────────────
# SIDEBAR  — Patient Input
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🫀</div>
        <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; color:#fff; margin-top:0.3rem;'>
            Patient Profile
        </div>
        <div style='font-family:"DM Mono",monospace; font-size:0.7rem; color:#8ab4d4;
                    letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem;'>
            Enter Clinical Values
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**DEMOGRAPHICS**")
    age      = st.slider("Age (years)",           29, 77, 54)
    sex      = st.radio("Sex", ["Male", "Female"], horizontal=True)
    sex_val  = 1 if sex == "Male" else 0

    st.markdown("<hr style='border-color:#1e2d4a; margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("**CARDIAC SYMPTOMS**")
    cp_map  = {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal Pain":2,"Asymptomatic":3}
    cp      = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    cp_val  = cp_map[cp]
    exang   = st.radio("Exercise Induced Angina", ["No","Yes"], horizontal=True)
    exang_v = 1 if exang == "Yes" else 0
    thalach = st.slider("Max Heart Rate (bpm)",   71, 202, 150)
    oldpeak = st.slider("ST Depression",          0.0, 6.2, 1.0, 0.1)
    slope_map = {"Downsloping":0,"Flat":1,"Upsloping":2}
    slope   = st.selectbox("ST Slope", list(slope_map.keys()))
    slope_v = slope_map[slope]

    st.markdown("<hr style='border-color:#1e2d4a; margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("**VITALS & LAB VALUES**")
    trestbps = st.slider("Resting BP (mmHg)",     94, 200, 130)
    chol     = st.slider("Cholesterol (mg/dl)",  126, 564, 240)
    fbs      = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"], horizontal=True)
    fbs_v    = 1 if fbs == "Yes" else 0

    st.markdown("<hr style='border-color:#1e2d4a; margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("**IMAGING & TESTS**")
    ecg_map  = {"Normal":0,"ST-T Abnormality":1,"Left Ventricular Hypertrophy":2}
    restecg  = st.selectbox("Resting ECG", list(ecg_map.keys()))
    ecg_v    = ecg_map[restecg]
    ca       = st.select_slider("Major Vessels (0–3)", options=[0,1,2,3], value=0)
    thal_map = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}
    thal     = st.selectbox("Thalassemia", list(thal_map.keys()))
    thal_v   = thal_map[thal]

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  RUN PREDICTION")


# ─────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0;'>Heart Disease Prediction</h1>
<p style='font-family:"DM Mono",monospace; font-size:0.8rem; color:#8ab4d4;
          letter-spacing:0.08em; text-transform:uppercase; margin-top:0.3rem;'>
    Decision Tree Classifier · Interpretable Clinical AI
</p>
<hr style='border:none; border-top:1px solid #1e2d4a; margin:1rem 0 1.5rem 0;'>
""", unsafe_allow_html=True)

# ── Build input array ─────────────────────────────────────────────
input_values = [age, sex_val, cp_val, trestbps, chol, fbs_v,
                ecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v]
input_array  = np.array([input_values])
input_df     = pd.DataFrame([input_values], columns=FEATURE_NAMES)

# ── Always show live prediction ───────────────────────────────────
pred      = model.predict(input_array)[0]
prob      = model.predict_proba(input_array)[0]
disease_p = prob[1]
no_dis_p  = prob[0]
path      = get_decision_path(model, input_array)

# ── Result Banner ─────────────────────────────────────────────────
if pred == 1:
    banner_class = "result-danger"
    icon  = "⚠️"
    title = "Heart Disease Detected"
    color = "#e74c3c"
else:
    banner_class = "result-safe"
    icon  = "✅"
    title = "No Heart Disease"
    color = "#2ecc71"

st.markdown(f"""
<div class="{banner_class}">
    <div style='font-size:2.8rem; margin-bottom:0.3rem;'>{icon}</div>
    <div class='result-subtitle' style='color:{color};'>Prediction Result</div>
    <div class='result-title' style='color:{color};'>{title}</div>
    <div class='result-subtitle' style='margin-top:0.5rem;'>
        {risk_label(disease_p)} &nbsp;·&nbsp; Confidence: {max(prob)*100:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Probability Metrics Row ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value' style='color:{color};'>{disease_p*100:.1f}%</div>
        <div class='metric-label'>Disease Probability</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value' style='color:#2ecc71;'>{no_dis_p*100:.1f}%</div>
        <div class='metric-label'>No Disease Probability</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value' style='color:#8ab4d4;'>{model.get_depth()}</div>
        <div class='metric-label'>Tree Depth</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value' style='color:#8ab4d4;'>{len(path)-1}</div>
        <div class='metric-label'>Decision Steps</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📜  Decision Path",
    "🌳  Tree Visualization",
    "📊  Feature Importance",
    "🧾  Patient Summary"
])


# ── TAB 1: Decision Path ──────────────────────────────────────────
with tab1:
    st.markdown("### How this prediction was made")
    st.markdown("""
    <p style='color:#8ab4d4; font-size:0.9rem;'>
    The tree evaluated each clinical feature in sequence.
    Each step narrows the diagnosis until a final leaf node is reached.
    </p>
    """, unsafe_allow_html=True)

    for i, step in enumerate(path):
        if step.get("leaf"):
            leaf_cls = "rule-leaf-danger" if step["cls_idx"] == 1 else "rule-leaf-safe"
            st.markdown(f"""
            <div class='rule-step {leaf_cls}'>
                🏁 &nbsp; <strong>FINAL LEAF</strong> &nbsp;→&nbsp;
                <strong>{step['class']}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {step['confidence']:.1f}%
                &nbsp;&nbsp;|&nbsp;&nbsp; Samples in node: {step['samples']}
            </div>
            """, unsafe_allow_html=True)
        else:
            side_css = "rule-step-left" if step["side"] == "left" else "rule-step-right"
            arrow    = "←" if step["side"] == "left" else "→"
            st.markdown(f"""
            <div class='rule-step {side_css}'>
                <span style='color:#8ab4d4;'>Step {i+1}</span>
                &nbsp;&nbsp;
                <strong style='color:#fff;'>{step['feature']}</strong>
                &nbsp;=&nbsp; <span style='color:#f1c40f;'>{step['value']:.1f}</span>
                &nbsp;&nbsp;
                <span style='color:#555;'>threshold: {step['threshold']:.2f}</span>
                &nbsp;&nbsp;
                <span style='color:#aaa;'>{arrow} Go {step['direction']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Probability Breakdown**")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        bars = ax.barh(['No Disease', 'Heart Disease'],
                       [no_dis_p*100, disease_p*100],
                       color=['#2ecc71', '#e74c3c'], height=0.5,
                       edgecolor='none')
        for bar, val in zip(bars, [no_dis_p*100, disease_p*100]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white',
                    fontsize=11, fontweight='bold',
                    fontfamily='monospace')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probability (%)', color='#8ab4d4', fontsize=9)
        ax.tick_params(colors='#8ab4d4', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d4a')
        ax.grid(axis='x', color='#1e2d4a', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── TAB 2: Tree Visualization ─────────────────────────────────────
with tab2:
    st.markdown("### Decision Tree Structure")
    st.markdown("""
    <p style='color:#8ab4d4; font-size:0.9rem;'>
    Orange nodes → predicts Heart Disease &nbsp;|&nbsp;
    Blue nodes → predicts No Disease &nbsp;|&nbsp;
    Darker shade = purer node
    </p>
    """, unsafe_allow_html=True)

    depth_choice = st.radio("Display depth", [3, 4, 5, "Full"], horizontal=True, index=0)
    show_depth   = None if depth_choice == "Full" else int(depth_choice)

    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')
    plot_tree(
        model,
        feature_names = FEATURE_LABELS,
        class_names   = CLASS_NAMES,
        filled        = True,
        rounded       = True,
        fontsize      = 8,
        max_depth     = show_depth,
        ax            = ax
    )
    ax.set_title(
        f"Heart Disease Decision Tree  (depth shown: {show_depth or 'full'})",
        color='#c8d8f0', fontsize=13,
        fontfamily='serif', pad=12
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄  View Raw Text Rules"):
        rules = export_text(model, feature_names=FEATURE_LABELS,
                            max_depth=6, spacing=3, show_weights=True)
        st.code(rules, language="text")


# ── TAB 3: Feature Importance ─────────────────────────────────────
with tab3:
    st.markdown("### Feature Importance Analysis")
    st.markdown("""
    <p style='color:#8ab4d4; font-size:0.9rem;'>
    Importance is measured by the total reduction in Gini impurity
    each feature contributes across all splits.
    </p>
    """, unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature':    FEATURE_LABELS,
        'Importance': importances,
        'Pct':        importances * 100
    }).sort_values('Importance', ascending=True)

    top5_min = fi_df['Importance'].nlargest(5).min()
    bar_colors = ['#e74c3c' if v >= top5_min else '#2c4a6e'
                  for v in fi_df['Importance']]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    bars = ax.barh(fi_df['Feature'], fi_df['Pct'],
                   color=bar_colors, edgecolor='none', height=0.65)
    for bar, val in zip(bars, fi_df['Pct']):
        if val > 0.3:
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', color='#c8d8f0',
                    fontsize=8.5, fontfamily='monospace')

    ax.set_xlabel('Importance (%)', color='#8ab4d4', fontsize=10)
    ax.tick_params(colors='#8ab4d4', labelsize=9)
    ax.set_xlim(0, fi_df['Pct'].max() * 1.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2d4a')
    ax.grid(axis='x', color='#1e2d4a', alpha=0.6)

    red_p  = mpatches.Patch(color='#e74c3c', label='Top 5 Features')
    blue_p = mpatches.Patch(color='#2c4a6e', label='Other Features')
    ax.legend(handles=[red_p, blue_p], facecolor='#111827',
              edgecolor='#1e2d4a', labelcolor='#8ab4d4', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top 5 table
    st.markdown("<br>", unsafe_allow_html=True)
    top5 = fi_df.sort_values('Importance', ascending=False).head(5).reset_index(drop=True)
    top5.index += 1
    top5['Importance (%)'] = top5['Pct'].apply(lambda x: f"{x:.3f}%")
    st.dataframe(
        top5[['Feature','Importance (%)']],
        use_container_width=True
    )

    # Current patient's feature values highlighted
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**This Patient's Feature Values vs Feature Importance**")

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor('#111827')
    ax2.set_facecolor('#111827')

    sorted_labels = fi_df['Feature'].tolist()
    sorted_imps   = fi_df['Importance'].tolist()

    # normalize patient values for overlay
    patient_vals_ordered = []
    for lbl in sorted_labels:
        idx = FEATURE_LABELS.index(lbl)
        patient_vals_ordered.append(input_values[idx])

    ax2.barh(sorted_labels, [i*100 for i in sorted_imps],
             color='#2c4a6e', edgecolor='none', height=0.65, label='Importance %')

    # Normalised patient value dots
    max_imp = max(sorted_imps) * 100
    for i, (lbl, pv) in enumerate(zip(sorted_labels, patient_vals_ordered)):
        norm_pv = min(pv / 300 * max_imp, max_imp)
        ax2.plot(norm_pv, i, 'o', color='#f39c12', markersize=7, zorder=5)

    ax2.set_xlabel('Importance (%)', color='#8ab4d4', fontsize=10)
    ax2.tick_params(colors='#8ab4d4', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#1e2d4a')
    ax2.grid(axis='x', color='#1e2d4a', alpha=0.6)
    imp_patch = mpatches.Patch(color='#2c4a6e', label='Feature Importance')
    dot_patch = mpatches.Patch(color='#f39c12', label='Patient Value (scaled)')
    ax2.legend(handles=[imp_patch, dot_patch], facecolor='#111827',
               edgecolor='#1e2d4a', labelcolor='#8ab4d4', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ── TAB 4: Patient Summary ────────────────────────────────────────
with tab4:
    st.markdown("### Patient Clinical Summary")

    data_display = {
        "Age":                  f"{age} years",
        "Sex":                  sex,
        "Chest Pain Type":      cp,
        "Resting BP":           f"{trestbps} mmHg",
        "Cholesterol":          f"{chol} mg/dl",
        "Fasting Blood Sugar":  fbs,
        "Resting ECG":          restecg,
        "Max Heart Rate":       f"{thalach} bpm",
        "Exercise Angina":      exang,
        "ST Depression":        f"{oldpeak}",
        "ST Slope":             slope,
        "Major Vessels":        f"{ca}",
        "Thalassemia":          thal
    }

    # Risk flags
    flags = []
    if age > 55:     flags.append(("Age > 55", "#e74c3c"))
    if sex_val == 1: flags.append(("Male Sex", "#f39c12"))
    if cp_val == 0:  flags.append(("Typical Angina", "#e74c3c"))
    if trestbps > 140: flags.append(("High Resting BP", "#f39c12"))
    if chol > 240:   flags.append(("High Cholesterol", "#f39c12"))
    if exang_v == 1: flags.append(("Exercise Angina", "#e74c3c"))
    if oldpeak > 1.5: flags.append(("High ST Depression", "#e74c3c"))
    if ca > 0:       flags.append(("Vessels Affected", "#e74c3c"))
    if thal_v == 3:  flags.append(("Reversible Thal Defect", "#e74c3c"))

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("**Clinical Values**")
        rows = list(data_display.items())
        for label, val in rows:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding:0.45rem 0.8rem; border-bottom:1px solid #1e2d4a;
                        font-size:0.88rem;'>
                <span style='color:#8ab4d4; font-family:"DM Mono",monospace;
                             font-size:0.78rem; text-transform:uppercase;
                             letter-spacing:0.05em;'>{label}</span>
                <span style='color:#e8eaf0; font-weight:500;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Risk Flags**")
        if flags:
            for flag, fc in flags:
                st.markdown(f"""
                <div style='background:#111827; border:1px solid {fc}33;
                            border-left:3px solid {fc}; border-radius:6px;
                            padding:0.4rem 0.8rem; margin:0.3rem 0;
                            font-family:"DM Mono",monospace; font-size:0.78rem;
                            color:{fc};'>
                    ⚠ {flag}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='color:#2ecc71; font-family:"DM Mono",monospace;
                        font-size:0.85rem; padding:1rem;'>
                ✅ No major risk flags detected
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Final Verdict**")
        verdict_color = "#e74c3c" if pred == 1 else "#2ecc71"
        st.markdown(f"""
        <div style='background:{"#2d0a0a" if pred==1 else "#0a2d14"};
                    border:1px solid {verdict_color};
                    border-radius:10px; padding:1.2rem; text-align:center;'>
            <div style='font-family:"DM Serif Display",serif;
                        font-size:1.6rem; color:{verdict_color};'>
                {"⚠️ Heart Disease" if pred==1 else "✅ No Heart Disease"}
            </div>
            <div style='font-family:"DM Mono",monospace; font-size:0.75rem;
                        color:#8ab4d4; margin-top:0.4rem;'>
                Disease probability: {disease_p*100:.1f}% &nbsp;|&nbsp; {risk_label(disease_p)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none; border-top:1px solid #1e2d4a; margin-top:3rem;'>
<p style='text-align:center; font-family:"DM Mono",monospace; font-size:0.72rem;
          color:#3a4a6a; letter-spacing:0.06em;'>
    HEART DISEASE PREDICTOR · DECISION TREE CLASSIFIER ·
    FOR EDUCATIONAL & RESEARCH USE ONLY · NOT FOR CLINICAL DIAGNOSIS
</p>
""", unsafe_allow_html=True)
