import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── LOAD MODEL ───────────────────────────────
model = joblib.load('outputs/xgb_model.pkl')
columns = joblib.load('outputs/model_columns.pkl')

st.title("🏦 Credit Risk Prediction System")
st.markdown("Enter borrower details to predict default risk")

# ── USER INPUTS ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    loan_amnt      = st.number_input("Loan Amount ($)", 1000, 40000, 10000)
    annual_inc     = st.number_input("Annual Income ($)", 10000, 500000, 60000)
    dti            = st.slider("Debt-to-Income Ratio %", 0.0, 50.0, 15.0)
    fico_range_low = st.slider("FICO Score", 580, 850, 700)
    int_rate       = st.slider("Interest Rate %", 5.0, 30.0, 13.0)

with col2:
    grade          = st.selectbox("Loan Grade", [0,1,2,3,4,5,6], 
          format_func=lambda x: 'ABCDEFG'[x])
    term           = st.selectbox("Term", [0,1], 
         format_func=lambda x: '36 months' if x==0 else '60 months')
    emp_length     = st.slider("Employment Length (years)", 0, 10, 3)
    home_ownership = st.selectbox("Home Ownership", [0,1,2], 
        format_func=lambda x: ['Rent','Own','Mortgage'][x])
    pub_rec        = st.number_input("Public Records", 0, 10, 0)

# ── PREDICT ──────────────────────────────────
if st.button("🔍 Predict Risk"):
    input_data = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': 0,
        'purpose': 2,
        'addr_state': 38,
        'dti': dti,
        'delinq_2yrs': 0,
        'fico_range_low': fico_range_low,
        'mths_since_last_delinq': 999,
        'open_acc': 8,
        'pub_rec': pub_rec,
        'revol_bal': 10000,
        'revol_util': 40.0,
        'total_acc': 20,
        'credit_score_avg': fico_range_low,
        'payment_to_income': loan_amnt / (annual_inc / 12),
        'high_dti_flag': int(dti > 35)
    }])

    input_data = input_data[columns]
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.15:
        risk = "🟢 Low Risk"
        color = "success"
        action = "Recommend APPROVAL"
    elif prob < 0.35:
        risk = "🟡 Medium Risk"
        color = "warning"
        action = "Recommend REVIEW"
    else:
        risk = "🔴 High Risk"
        color = "error"
        action = "Recommend DECLINE"

    st.markdown("---")
    st.subheader("Prediction Result")
    getattr(st, color)(f"{risk} — {prob:.1%} probability of default")
    st.info(f"💼 Bank Action: {action}")

    st.metric("Default Probability", f"{prob:.1%}")