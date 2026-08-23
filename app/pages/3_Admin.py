import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Admin | EMIPredict AI", page_icon="🗂️", layout="wide")

ADMIN_DATA_PATH = "data/admin_records.csv"

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
    font-size: 1.2rem;
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

/* Form styling matching Home.py */
div[data-testid="stForm"] {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 16px;
    padding: 1.6rem;
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {
    background-color: #0B1E33 !important;
    border: 1px solid #274863 !important;
    border-radius: 8px !important;
    color: #E8EEF4 !important;
}
label {
    color: #C4D3E0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stFormSubmitButton button {
    background: linear-gradient(120deg, #2DD4BF, #14B8A6) !important;
    color: #0B1E33 !important;
    font-family: 'Sora', sans-serif;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}
.stFormSubmitButton button:hover {
    background: linear-gradient(120deg, #14B8A6, #0F9E8F) !important;
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
    background: #10233A;
    border: 1px solid #1F3A56;
    border-radius: 12px;
    padding: 0.5rem;
}

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
/* Delete/danger button variant inside the main content area */
div[data-testid="stButton"] button[kind="secondary"]:has(div:contains("Delete")) {
    border-color: #EF5350 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR BRANDING + CUSTOM NAVIGATION
# ============================================================
CURRENT_PAGE = "admin"

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
    <div class="emi-eyebrow">Data Management</div>
    <p class="emi-title">Admin Panel</p>
    <p class="emi-subtitle">View, add, edit, and delete customer application records.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD / INITIALIZE ADMIN RECORDS (separate working copy, persisted to CSV)
# ============================================================
COLUMNS = [
    "customer_name", "age", "gender", "monthly_salary", "credit_score",
    "emi_scenario", "requested_amount", "requested_tenure", "emi_eligibility"
]

def load_records():
    if os.path.exists(ADMIN_DATA_PATH):
        return pd.read_csv(ADMIN_DATA_PATH)
    else:
        # seed with a small sample from the main dataset so the table isn't empty on first run
        base = pd.read_csv("data/eda_sample.csv", low_memory=False).head(15)
        seeded = pd.DataFrame({
            "customer_name": [f"Customer {i+1}" for i in range(len(base))],
            "age": base["age"].astype(int),
            "gender": base["gender"],
            "monthly_salary": base["monthly_salary"],
            "credit_score": base["credit_score"].astype(int),
            "emi_scenario": base["emi_scenario"],
            "requested_amount": base["requested_amount"],
            "requested_tenure": base["requested_tenure"],
            "emi_eligibility": base["emi_eligibility"],
        })
        seeded.to_csv(ADMIN_DATA_PATH, index=False)
        return seeded

if "admin_df" not in st.session_state:
    st.session_state.admin_df = load_records()

def save_records():
    st.session_state.admin_df.to_csv(ADMIN_DATA_PATH, index=False)

# ============================================================
# VIEW RECORDS
# ============================================================
st.markdown('<div class="section-title">Customer Records</div>', unsafe_allow_html=True)
st.dataframe(st.session_state.admin_df, width="stretch", height=350)

st.caption(f"{len(st.session_state.admin_df)} records total")

# ============================================================
# ADD RECORD
# ============================================================
st.markdown('<div class="section-title">Add New Record</div>', unsafe_allow_html=True)

with st.form("add_record_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        customer_name = st.text_input("Customer Name")
        age = st.number_input("Age", min_value=18, max_value=75, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        emi_scenario = st.selectbox("EMI Scenario", [
            "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
            "Personal Loan EMI", "Education EMI"
        ])
    with c3:
        requested_amount = st.number_input("Requested Amount (INR)", min_value=1000, value=100000, step=5000)
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=24)
        emi_eligibility = st.selectbox("Eligibility", ["Eligible", "High_Risk", "Not_Eligible"])

    add_submitted = st.form_submit_button("Add Record", width="stretch")

    if add_submitted:
        if not customer_name.strip():
            st.error("Customer name is required.")
        else:
            new_row = pd.DataFrame([{
                "customer_name": customer_name.strip(),
                "age": age, "gender": gender, "monthly_salary": monthly_salary,
                "credit_score": credit_score, "emi_scenario": emi_scenario,
                "requested_amount": requested_amount, "requested_tenure": requested_tenure,
                "emi_eligibility": emi_eligibility,
            }])
            st.session_state.admin_df = pd.concat([st.session_state.admin_df, new_row], ignore_index=True)
            save_records()
            st.success(f"Added record for {customer_name.strip()}.")
            st.rerun()

# ============================================================
# EDIT / DELETE RECORD
# ============================================================
st.markdown('<div class="section-title">Edit or Delete a Record</div>', unsafe_allow_html=True)

if len(st.session_state.admin_df) == 0:
    st.info("No records available to edit or delete.")
else:
    options = [
        f"{i} — {row['customer_name']} ({row['emi_scenario']})"
        for i, row in st.session_state.admin_df.iterrows()
    ]
    selected = st.selectbox("Select a record", options)
    selected_idx = int(selected.split(" — ")[0])
    record = st.session_state.admin_df.loc[selected_idx]

    with st.form("edit_record_form"):
        e1, e2, e3 = st.columns(3)
        with e1:
            edit_name = st.text_input("Customer Name", value=str(record["customer_name"]))
            edit_age = st.number_input("Age", min_value=18, max_value=75, value=int(record["age"]))
            edit_gender = st.selectbox("Gender", ["Male", "Female"],
                                        index=["Male", "Female"].index(record["gender"]))
        with e2:
            edit_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=int(record["monthly_salary"]), step=1000)
            edit_credit = st.number_input("Credit Score", min_value=300, max_value=900, value=int(record["credit_score"]))
            scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
                                 "Personal Loan EMI", "Education EMI"]
            edit_scenario = st.selectbox("EMI Scenario", scenario_options,
                                          index=scenario_options.index(record["emi_scenario"]))
        with e3:
            edit_amount = st.number_input("Requested Amount (INR)", min_value=1000, value=int(record["requested_amount"]), step=5000)
            edit_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=int(record["requested_tenure"]))
            elig_options = ["Eligible", "High_Risk", "Not_Eligible"]
            edit_eligibility = st.selectbox("Eligibility", elig_options,
                                             index=elig_options.index(record["emi_eligibility"]))

        b1, b2 = st.columns(2)
        with b1:
            update_submitted = st.form_submit_button("Update Record", width="stretch")
        with b2:
            delete_submitted = st.form_submit_button("Delete Record", width="stretch")

        if update_submitted:
            st.session_state.admin_df.loc[selected_idx] = [
                edit_name.strip(), edit_age, edit_gender, edit_salary,
                edit_credit, edit_scenario, edit_amount, edit_tenure, edit_eligibility
            ]
            save_records()
            st.success(f"Updated record for {edit_name.strip()}.")
            st.rerun()

        if delete_submitted:
            st.session_state.admin_df = st.session_state.admin_df.drop(index=selected_idx).reset_index(drop=True)
            save_records()
            st.success("Record deleted.")
            st.rerun()