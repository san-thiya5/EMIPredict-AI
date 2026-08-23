import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="EDA | EMIPredict AI", page_icon="📊", layout="wide")

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
.chart-card {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}
.chart-title {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #2DD4BF;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.insight-box {
    background: #0F2438;
    border-left: 3px solid #2DD4BF;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.92rem;
    color: #C4D3E0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B1826;
    border-right: 1px solid #1F3A56;
}

/* Force-hide Streamlit's automatic page nav (multiple selector variants for version safety) */
section[data-testid="stSidebarNav"],
div[data-testid="stSidebarNav"],
ul[data-testid="stSidebarNavItems"],
div[data-testid="stSidebarNavItems"],
nav[aria-label="Page navigation"] {
    display: none !important;
    height: 0 !important;
}

/* Custom nav buttons styled as cards */
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
/* Inactive (secondary) nav button */
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
/* Active (primary) nav button */
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
CURRENT_PAGE = "eda"

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

st.markdown("""
<div class="emi-header">
    <div class="emi-eyebrow">Data Exploration</div>
    <p class="emi-title">Exploratory Data Analysis</p>
    <p class="emi-subtitle">Key patterns in the 400,800-record EMI dataset that shaped feature engineering and model design.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA (cached)
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/eda_sample.csv", low_memory=False)

df = load_data()

# ============================================================
# KEY STATS ROW
# ============================================================
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Total Records", f"{len(df):,}"),
    ("Eligible Rate", f"{(df['emi_eligibility'] == 'Eligible').mean()*100:.1f}%"),
    ("High Risk Rate", f"{(df['emi_eligibility'] == 'High_Risk').mean()*100:.1f}%"),
    ("Avg Max EMI", f"₹{df['max_monthly_emi'].mean():,.0f}"),
]
for col, (label, value) in zip([c1, c2, c3, c4], stats):
    with col:
        stat_html = (
            f'<div class="chart-card" style="text-align:center;">'
            f'<div style="color:#9FB3C8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">{label}</div>'
            f'<div style="font-family:\'Sora\',sans-serif; font-size:1.6rem; font-weight:800; color:#F4F8FB; margin-top:0.3rem;">{value}</div>'
            f'</div>'
        )
        st.markdown(stat_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# CHARTS - using the pre-generated PNGs from src/05_eda.py
# ============================================================
chart_files = {
    "Eligibility Distribution": "notebooks/eda_target_distribution.png",
    "Max Monthly EMI Distribution": "notebooks/eda_emi_distribution.png",
    "Eligibility Rate by EMI Scenario": "notebooks/eda_eligibility_by_scenario.png",
    "Feature Correlation Heatmap": "notebooks/eda_correlation_heatmap.png",
    "Credit Score by Eligibility Class": "notebooks/eda_credit_score_by_eligibility.png",
    "Salary vs Max Monthly EMI": "notebooks/eda_salary_vs_emi.png",
}

row1 = st.columns(2)
row2 = st.columns(2)
row3 = st.columns(2)
slots = row1 + row2 + row3

for slot, (title, path) in zip(slots, chart_files.items()):
    with slot:
        st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
        if os.path.exists(path):
            st.image(path, width="stretch")
        else:
            st.warning(f"Chart not found: {path}. Run src/05_eda.py first.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# KEY INSIGHTS
# ============================================================
insight_html = (
    '<div class="insight-box">'
    '<b>Key insights:</b><br>'
    '• Vehicle and Personal Loan EMIs have the lowest approval rates (~14% Eligible) versus E-commerce and Home Appliances (~26%) — scenario type is a strong predictor.<br>'
    '• Credit score separates eligibility classes only modestly (Eligible avg 725.6 vs Not_Eligible avg 694.1) — it works best combined with income/expense signals, not alone.<br>'
    '• Affordability (max_monthly_emi) correlates most strongly with spending/liquidity features (groceries & utilities, bank balance, travel expenses, emergency fund) — more than with raw monthly salary.'
    '</div>'
)
st.markdown(insight_html, unsafe_allow_html=True)