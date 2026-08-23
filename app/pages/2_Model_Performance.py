import streamlit as st

st.set_page_config(page_title="Model Performance | EMIPredict AI", page_icon="📈", layout="wide")

# ============================================================
# SHARED STYLING (same theme as Home.py)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #0B1E33 0%, #0E2338 100%);
    color: #E8EEF4;
}
.emi-header {
    padding: 2rem 2.5rem;
    background: linear-gradient(120deg, #132A45 0%, #0F2438 100%);
    border-radius: 16px;
    border: 1px solid #1F3A56;
    margin-bottom: 1.5rem;
}
.emi-eyebrow {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2DD4BF;
    margin-bottom: 0.4rem;
}
.emi-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #F4F8FB;
    margin: 0;
}
.emi-subtitle {
    font-size: 0.98rem;
    color: #9FB3C8;
    margin-top: 0.4rem;
}
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #F4F8FB;
    margin: 1.5rem 0 1rem 0;
}
.chart-card {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.winner-card {
    background: linear-gradient(120deg, rgba(45,212,191,0.12), rgba(45,212,191,0.03));
    border: 1px solid #2DD4BF;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.model-name {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #F4F8FB;
    margin-bottom: 0.9rem;
}
.winner-badge {
    display: inline-block;
    background: rgba(45, 212, 191, 0.18);
    color: #2DD4BF;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-left: 0.6rem;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(31, 58, 86, 0.5);
    font-size: 0.9rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-row-label { color: #9FB3C8; }
.metric-row-value { color: #E8EEF4; font-weight: 600; font-family: 'Sora', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B1826;
    border-right: 1px solid #1F3A56;
}
section[data-testid="stSidebarNav"],
div[data-testid="stSidebarNav"],
ul[data-testid="stSidebarNavItems"],
div[data-testid="stSidebarNavItems"],
nav[aria-label="Page navigation"] {
    display: none !important;
    height: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: 0.6rem;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
    background: #10233A !important;
    border: 1px solid #1F3A56 !important;
    color: #C4D3E0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #142A45 !important;
    border-color: #2DD4BF !important;
    color: #2DD4BF !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: rgba(45, 212, 191, 0.14) !important;
    border: 1px solid #2DD4BF !important;
    color: #2DD4BF !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR BRANDING + CUSTOM NAVIGATION
# ============================================================
CURRENT_PAGE = "performance"

st.sidebar.markdown("""
<div style="padding: 0.5rem 0.5rem 1rem 0.5rem; border-bottom: 1px solid #1F3A56; margin-bottom: 1rem;">
    <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.1rem; color:#F4F8FB;">💠 EMIPredict AI</div>
    <div style="font-size:0.72rem; color:#6B8299; letter-spacing:0.04em;">Risk Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🏠  Home", key="nav_home",
                      type="primary" if CURRENT_PAGE == "home" else "secondary"):
    if CURRENT_PAGE != "home":
        st.switch_page("Home.py")

if st.sidebar.button("📊  EDA", key="nav_eda",
                      type="primary" if CURRENT_PAGE == "eda" else "secondary"):
    if CURRENT_PAGE != "eda":
        st.switch_page("pages/1_EDA.py")

if st.sidebar.button("📈  Model Performance", key="nav_perf",
                      type="primary" if CURRENT_PAGE == "performance" else "secondary"):
    if CURRENT_PAGE != "performance":
        st.switch_page("pages/2_Model_Performance.py")

if st.sidebar.button("🗂️  Admin", key="nav_admin",
                      type="primary" if CURRENT_PAGE == "admin" else "secondary"):
    if CURRENT_PAGE != "admin":
        st.switch_page("pages/3_Admin.py")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="emi-header">
    <div class="emi-eyebrow">Model Evaluation</div>
    <p class="emi-title">Model Performance</p>
    <p class="emi-subtitle">Comparison of all trained models, tracked and logged via MLflow, on the held-out validation set.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CLASSIFICATION RESULTS (hardcoded from training run)
# ============================================================
st.markdown('<div class="section-title">Classification &mdash; EMI Eligibility</div>', unsafe_allow_html=True)

clf_results = {
    "XGBoost": {"accuracy": 0.9749, "precision": 0.9732, "recall": 0.9749, "f1": 0.9729, "auc": 0.9986, "winner": True},
    "RandomForest": {"accuracy": 0.8753, "precision": 0.9423, "recall": 0.8753, "f1": 0.9011, "auc": 0.9874, "winner": False},
    "LogisticRegression": {"accuracy": 0.7827, "precision": 0.9147, "recall": 0.7827, "f1": 0.8342, "auc": 0.9541, "winner": False},
}

clf_cols = st.columns(3)
for col, (name, m) in zip(clf_cols, clf_results.items()):
    with col:
        card_class = "winner-card" if m["winner"] else "chart-card"
        badge = '<span class="winner-badge">Best Model</span>' if m["winner"] else ""
        rows = "".join([
            f'<div class="metric-row"><span class="metric-row-label">Accuracy</span><span class="metric-row-value">{m["accuracy"]*100:.2f}%</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">Precision</span><span class="metric-row-value">{m["precision"]*100:.2f}%</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">Recall</span><span class="metric-row-value">{m["recall"]*100:.2f}%</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">F1 Score</span><span class="metric-row-value">{m["f1"]*100:.2f}%</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">ROC-AUC</span><span class="metric-row-value">{m["auc"]*100:.2f}%</span></div>',
        ])
        card_html = (
            f'<div class="{card_class}">'
            f'<div class="model-name">{name}{badge}</div>'
            f'{rows}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

# ============================================================
# REGRESSION RESULTS (hardcoded from training run)
# ============================================================
st.markdown('<div class="section-title">Regression &mdash; Max Monthly EMI</div>', unsafe_allow_html=True)

reg_results = {
    "XGBoost": {"rmse": 664.59, "mae": 222.69, "r2": 0.9926, "mape": 0.0746, "winner": True},
    "RandomForest": {"rmse": 899.04, "mae": 263.55, "r2": 0.9865, "mape": 0.0538, "winner": False},
    "LinearRegression": {"rmse": 4098.56, "mae": 2927.49, "r2": 0.7197, "mape": 1.9197, "winner": False},
}

reg_cols = st.columns(3)
for col, (name, m) in zip(reg_cols, reg_results.items()):
    with col:
        card_class = "winner-card" if m["winner"] else "chart-card"
        badge = '<span class="winner-badge">Best Model</span>' if m["winner"] else ""
        rows = "".join([
            f'<div class="metric-row"><span class="metric-row-label">RMSE</span><span class="metric-row-value">₹{m["rmse"]:,.2f}</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">MAE</span><span class="metric-row-value">₹{m["mae"]:,.2f}</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">R&sup2; Score</span><span class="metric-row-value">{m["r2"]:.4f}</span></div>',
            f'<div class="metric-row"><span class="metric-row-label">MAPE</span><span class="metric-row-value">{m["mape"]*100:.2f}%</span></div>',
        ])
        card_html = (
            f'<div class="{card_class}">'
            f'<div class="model-name">{name}{badge}</div>'
            f'{rows}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

# ============================================================
# TARGETS & SUMMARY
# ============================================================
st.markdown('<div class="section-title">Project Targets</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    target_html = (
        '<div class="chart-card">'
        '<div class="model-name">Classification Accuracy Target</div>'
        '<div class="metric-row"><span class="metric-row-label">Target</span><span class="metric-row-value">&gt; 90%</span></div>'
        '<div class="metric-row"><span class="metric-row-label">Achieved (XGBoost)</span><span class="metric-row-value" style="color:#2DD4BF;">97.49%</span></div>'
        '</div>'
    )
    st.markdown(target_html, unsafe_allow_html=True)
with t2:
    target_html2 = (
        '<div class="chart-card">'
        '<div class="model-name">Regression RMSE Target</div>'
        '<div class="metric-row"><span class="metric-row-label">Target</span><span class="metric-row-value">&lt; ₹2,000</span></div>'
        '<div class="metric-row"><span class="metric-row-label">Achieved (XGBoost)</span><span class="metric-row-value" style="color:#2DD4BF;">₹664.59</span></div>'
        '</div>'
    )
    st.markdown(target_html2, unsafe_allow_html=True)

st.markdown("""
<div class="chart-card" style="border-left: 3px solid #2DD4BF;">
<b>Summary:</b> XGBoost was selected as the production model for both tasks, comfortably exceeding both project targets.
All experiments (parameters, metrics, and model artifacts) were tracked using MLflow and registered in the model registry
as <code>EMI_Eligibility_Classifier</code> and <code>EMI_Amount_Regressor</code>.
</div>
""", unsafe_allow_html=True)