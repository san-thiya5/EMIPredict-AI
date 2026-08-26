import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="EMIPredict AI", page_icon="💠", layout="wide")

# ============================================================
# CUSTOM STYLING
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

/* Header banner */
.emi-header {
    padding: 2rem 2.5rem;
    background: linear-gradient(120deg, #132A45 0%, #0F2438 100%);
    border-radius: 16px;
    border: 1px solid #1F3A56;
    margin-bottom: 1.5rem;
}
.emi-eyebrow {
    font-family: 'Inter', sans-serif;
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
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: #9FB3C8;
    margin-top: 0.4rem;
}

/* Section labels inside form */
.emi-section-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2DD4BF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid #1F3A56;
    padding-bottom: 0.6rem;
}

/* Card container for the form */
div[data-testid="stForm"] {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 16px;
    padding: 2rem;
}

/* Give each of the 3 form columns its own padded card, with a gap between them */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    gap: 1.75rem;
}
div[data-testid="stForm"] div[data-testid="column"] {
    background: #0E1F35;
    border: 1px solid #1B3350;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
}

/* Space out each input field vertically */
div[data-testid="stForm"] div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.85rem;
}

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #0B1E33 !important;
    border: 1px solid #274863 !important;
    border-radius: 8px !important;
    color: #E8EEF4 !important;
}
label {
    color: #C4D3E0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    margin-bottom: 0.3rem !important;
}

/* Submit button */
.stFormSubmitButton button {
    background: linear-gradient(120deg, #2DD4BF, #14B8A6) !important;
    color: #0B1E33 !important;
    font-family: 'Sora', sans-serif;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 0 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
}
.stFormSubmitButton button:hover {
    background: linear-gradient(120deg, #14B8A6, #0F9E8F) !important;
}

/* Result cards */
.result-card {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
}
.status-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.status-eligible { background: rgba(45, 212, 191, 0.15); color: #2DD4BF; border: 1px solid #2DD4BF; }
.status-highrisk { background: rgba(245, 166, 35, 0.15); color: #F5A623; border: 1px solid #F5A623; }
.status-noteligible { background: rgba(239, 83, 80, 0.15); color: #EF5350; border: 1px solid #EF5350; }

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9FB3C8;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #F4F8FB;
    margin-bottom: 0.3rem;
}
.metric-sub {
    font-size: 0.9rem;
    color: #9FB3C8;
}

.affordability-ok { color: #2DD4BF; font-weight: 600; }
.affordability-over { color: #F5A623; font-weight: 600; }

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
CURRENT_PAGE = "home"

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
# LOAD MODELS AND FEATURE LISTS (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_models():
    clf = xgb.XGBClassifier()
    clf.load_model("app/models/emi_eligibility_classifier.json")

    reg = xgb.XGBRegressor()
    reg.load_model("app/models/emi_amount_regressor.json")

    with open("app/models/classifier_features.txt") as f:
        clf_features = f.read().splitlines()

    with open("app/models/regressor_features.txt") as f:
        reg_features = f.read().splitlines()

    return clf, reg, clf_features, reg_features

clf_model, reg_model, clf_features, reg_features = load_models()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="emi-header">
    <div class="emi-eyebrow">Financial Risk Intelligence</div>
    <p class="emi-title">EMIPredict AI</p>
    <p class="emi-subtitle">Instant EMI eligibility scoring and affordability estimation, powered by gradient-boosted risk models.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM (raw, human-friendly fields)
# ============================================================
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="emi-section-label">Personal Details</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=75, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Professional"])
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)

    with col2:
        st.markdown('<div class="emi-section-label">Employment & Income</div>', unsafe_allow_html=True)
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0, step=0.5)
        company_type = st.selectbox("Company Type", ["Government", "Private", "MNC", "Startup"])
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, value=100000, step=5000)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, value=50000, step=5000)

    with col3:
        st.markdown('<div class="emi-section-label">Expenses & Existing Loans</div>', unsafe_allow_html=True)
        house_type = st.selectbox("House Type", ["Owned", "Rented"])
        monthly_rent = st.number_input("Monthly Rent (INR, 0 if owned)", min_value=0, value=0, step=500)
        school_fees = st.number_input("School Fees (INR/month)", min_value=0, value=0, step=500)
        college_fees = st.number_input("College Fees (INR/month)", min_value=0, value=0, step=500)
        travel_expenses = st.number_input("Travel Expenses (INR/month)", min_value=0, value=3000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (INR/month)", min_value=0, value=8000, step=500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", min_value=0, value=2000, step=500)
        existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, value=0, step=500)

    st.markdown('<div style="margin-top:1.8rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="emi-section-label">Requested EMI</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
             "Personal Loan EMI", "Education EMI"]
        )
        requested_amount = st.number_input("Requested Amount (INR)", min_value=1000, value=200000, step=5000)
    with col5:
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=24)

    submitted = st.form_submit_button("Check Eligibility", width="stretch")

# ============================================================
# FEATURE ENGINEERING (mirrors src/06_feature_engineering.py exactly)
# ============================================================
def build_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    expense_cols = ["monthly_rent", "school_fees", "college_fees",
                     "travel_expenses", "groceries_utilities", "other_monthly_expenses"]
    df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)

    df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    df["debt_to_income"] = df["debt_to_income"].fillna(0)

    df["expense_to_income"] = df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
    df["expense_to_income"] = df["expense_to_income"].fillna(0)

    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"] - df["current_emi_amount"]

    df["requested_to_disposable"] = df["requested_amount"] / df["disposable_income"].replace(0, np.nan)
    df["requested_to_disposable"] = df["requested_to_disposable"].fillna(0)
    df["requested_to_disposable"] = df["requested_to_disposable"].clip(-500, 500)

    df["savings_to_income"] = (df["emergency_fund"] + df["bank_balance"]) / df["monthly_salary"].replace(0, np.nan)
    df["savings_to_income"] = df["savings_to_income"].fillna(0)

    df["is_stable_employment"] = (df["years_of_employment"] >= 2).astype(int)

    df["existing_loans_flag"] = df["existing_loans"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["existing_loans"])

    categorical_cols = ["gender", "marital_status", "education", "employment_type",
                         "company_type", "house_type", "emi_scenario"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df_encoded

def align_features(df_encoded: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[expected_features]

# ============================================================
# PREDICTION + RESULTS
# ============================================================
if submitted:
    raw_input = {
        "age": age, "gender": gender, "marital_status": marital_status,
        "education": education, "monthly_salary": monthly_salary,
        "employment_type": employment_type, "years_of_employment": years_of_employment,
        "company_type": company_type, "house_type": house_type, "monthly_rent": monthly_rent,
        "family_size": family_size, "dependents": dependents, "school_fees": school_fees,
        "college_fees": college_fees, "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities, "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
        "credit_score": credit_score, "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    }

    engineered = build_features(raw_input)
    X_clf = align_features(engineered.copy(), clf_features)
    X_reg = align_features(engineered.copy(), reg_features)

    eligibility_pred = clf_model.predict(X_clf)[0]
    eligibility_probs = clf_model.predict_proba(X_clf)[0]
    label_map = {0: "Not_Eligible", 1: "High_Risk", 2: "Eligible"}
    eligibility_label = label_map[eligibility_pred]
    confidence = eligibility_probs[eligibility_pred] * 100

    max_emi_pred = reg_model.predict(X_reg)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns(2)

    pill_class = {"Eligible": "status-eligible", "High_Risk": "status-highrisk", "Not_Eligible": "status-noteligible"}
    pill_text = {"Eligible": "✓ Eligible", "High_Risk": "⚠ High Risk", "Not_Eligible": "✕ Not Eligible"}

    with res_col1:
        class_names = ["Not_Eligible", "High_Risk", "Eligible"]
        class_colors = {"Not_Eligible": "#EF5350", "High_Risk": "#F5A623", "Eligible": "#2DD4BF"}

        bars_html = ""
        for i, cname in enumerate(class_names):
            pct = eligibility_probs[i] * 100
            is_predicted = (cname == eligibility_label)
            bar_color = class_colors[cname]
            opacity = "1" if is_predicted else "0.45"
            weight = "700" if is_predicted else "500"
            bars_html += (
                f'<div style="margin-bottom: 0.7rem;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#C4D3E0; font-weight:{weight}; margin-bottom:0.25rem;">'
                f'<span>{cname}</span><span>{pct:.1f}%</span>'
                f'</div>'
                f'<div style="background:#0B1E33; border-radius:6px; height:10px; overflow:hidden;">'
                f'<div style="width:{pct}%; background:{bar_color}; opacity:{opacity}; height:100%; border-radius:6px;"></div>'
                f'</div>'
                f'</div>'
            )

        card_html = (
            f'<div class="result-card">'
            f'<div class="status-pill {pill_class[eligibility_label]}">{pill_text[eligibility_label]}</div>'
            f'<div class="metric-label">Model Confidence</div>'
            f'<div class="metric-value">{confidence:.1f}%</div>'
            f'<div style="margin-top:1.2rem;">{bars_html}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with res_col2:
        implied_emi = requested_amount / requested_tenure if requested_tenure > 0 else 0
        if implied_emi <= max_emi_pred:
            afford_note = '<span class="affordability-ok">Within your affordable range</span>'
        else:
            afford_note = '<span class="affordability-over">Exceeds your affordable range</span>'

        card2_html = (
            f'<div class="result-card">'
            f'<div class="metric-label">Max Affordable Monthly EMI</div>'
            f'<div class="metric-value">₹{max_emi_pred:,.0f}</div>'
            f'<div class="metric-sub">Your requested EMI: ₹{implied_emi:,.0f}/month &nbsp;·&nbsp; {afford_note}</div>'
            f'</div>'
        )
        st.markdown(card2_html, unsafe_allow_html=True)