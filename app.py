
# IMPORTS

import streamlit as st
import joblib
import numpy as np
import time


# PAGE CONFIG

st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    background-color: #0E4C92;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL

model = joblib.load("project_1.pkl")


# TITLE

st.title("🏦 CreditWise Loan Approval System")
st.markdown("### Smart AI-based Loan Risk Assessment")
st.markdown("---")


# INPUT SECTIONS


# ---- Applicant Info ----
st.markdown("## 👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    Applicant_Income = st.number_input("💰 Applicant Income", min_value=0.0)

with col2:
    Coapplicant_Income = st.number_input("💵 Coapplicant Income", min_value=0.0)

with col3:
    Age = st.number_input("🎂 Age", min_value=18)

col4, col5, col6 = st.columns(3)

with col4:
    Dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0)

with col5:
    Existing_Loans = st.number_input("🏦 Existing Loans", min_value=0)

with col6:
    Savings = st.number_input("💎 Savings", min_value=0.0)

# ---- Loan Info ----
st.markdown("## 🏠 Loan Details")

col7, col8, col9 = st.columns(3)

with col7:
    Loan_Amount = st.number_input("💳 Loan Amount", min_value=0.0)

with col8:
    Loan_Term = st.number_input("📆 Loan Term (months)", min_value=1)

with col9:
    Collateral_Value = st.number_input("🏡 Collateral Value", min_value=0.0)

# ---- Credit Info ----
st.markdown("## 📊 Credit Details")

Credit_Score = st.number_input("📈 Credit Score", min_value=0.0)

# Categorical 
st.markdown("## 📄 Employment & Personal Details")

col10, col11, col12 = st.columns(3)

with col10:
    Education_Level = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])

with col11:
    Employment_Status = st.selectbox(
        "💼 Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

with col12:
    Marital_Status = st.selectbox("💍 Marital Status", ["Single", "Married"])

col13, col14, col15 = st.columns(3)

with col13:
    Loan_Purpose = st.selectbox(
        "🎯 Loan Purpose",
        ["Car", "Education", "Home", "Personal"]
    )

with col14:
    Property_Area = st.selectbox(
        "📍 Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

with col15:
    Gender = st.selectbox("🧑 Gender", ["Male", "Female"])

Employer_Category = st.selectbox(
    "🏢 Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)


# FEATURE ENGINEERING


DTI_Ratio = Loan_Amount / (Applicant_Income + 1)
DTI_Ratio_sq = DTI_Ratio ** 2
Credit_Score_sq = Credit_Score ** 2


# ONE HOT ENCODING


Employment_Status_Salaried = 1 if Employment_Status == "Salaried" else 0
Employment_Status_Self_employed = 1 if Employment_Status == "Self-employed" else 0
Employment_Status_Unemployed = 1 if Employment_Status == "Unemployed" else 0

Marital_Status_Single = 1 if Marital_Status == "Single" else 0

Loan_Purpose_Car = 1 if Loan_Purpose == "Car" else 0
Loan_Purpose_Education = 1 if Loan_Purpose == "Education" else 0
Loan_Purpose_Home = 1 if Loan_Purpose == "Home" else 0
Loan_Purpose_Personal = 1 if Loan_Purpose == "Personal" else 0

Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
Property_Area_Urban = 1 if Property_Area == "Urban" else 0

Gender_Male = 1 if Gender == "Male" else 0

Employer_Category_Government = 1 if Employer_Category == "Government" else 0
Employer_Category_MNC = 1 if Employer_Category == "MNC" else 0
Employer_Category_Private = 1 if Employer_Category == "Private" else 0
Employer_Category_Unemployed = 1 if Employer_Category == "Unemployed" else 0

Education_Level_Binary = 1 if Education_Level == "Graduate" else 0


# SIDEBAR PREDICTION


with st.sidebar:
    st.header("🔍 Prediction Control")
    predict_btn = st.button("🚀 Predict Loan Status")


# PREDICTION


if predict_btn:

    input_data = np.array([[

        Applicant_Income,
        Coapplicant_Income,
        Age,
        Dependents,
        Existing_Loans,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        Education_Level_Binary,
        Employment_Status_Salaried,
        Employment_Status_Self_employed,
        Employment_Status_Unemployed,
        Marital_Status_Single,
        Loan_Purpose_Car,
        Loan_Purpose_Education,
        Loan_Purpose_Home,
        Loan_Purpose_Personal,
        Property_Area_Semiurban,
        Property_Area_Urban,
        Gender_Male,
        Employer_Category_Government,
        Employer_Category_MNC,
        Employer_Category_Private,
        Employer_Category_Unemployed,
        DTI_Ratio_sq,
        Credit_Score_sq

    ]])

    with st.spinner("Analyzing application..."):
        time.sleep(2)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric("Approval Probability", f"{probability*100:.2f}%")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")