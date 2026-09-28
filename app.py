import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Review", page_icon="🏦", layout="wide")


# ── LOAD MODEL ───────────────────────────────
# cache so the model loads once, not on every click
@st.cache_resource
def load_files():
    model = joblib.load('outputs/xgb_model.pkl')
    columns = joblib.load('outputs/model_columns.pkl')
    encoders = joblib.load('outputs/encoders.pkl')
    cutoffs = joblib.load('outputs/risk_cutoffs.pkl')
    return model, columns, encoders, cutoffs

model, columns, encoders, cutoffs = load_files()


# ── PAGE STYLE ───────────────────────────────
# plain, form-like look, closer to an internal bank tool
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

.stApp { background: #F3F5F8; }
.stApp, .stApp p, .stApp label, .stApp input, .stApp h1, .stApp h2, .stApp h3, .stApp button {
    font-family: 'Public Sans', 'Segoe UI', Arial, sans-serif;
}
h1 { color: #1F2A44; font-size: 1.8rem !important; font-weight: 700 !important; padding-bottom: 0 !important; }
.subtitle { color: #5B6475; margin: 2px 0 22px 0; }

div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #D9DEE7;
    border-radius: 6px;
    padding: 24px 28px;
}
.section {
    color: #1F2A44;
    font-weight: 600;
    font-size: 1.02rem;
    border-bottom: 1px solid #E3E7EE;
    padding-bottom: 6px;
    margin: 14px 0 8px 0;
}

.slip {
    background: #FFFFFF;
    border: 1px solid #D9DEE7;
    border-top: 6px solid var(--band);
    border-radius: 6px;
    padding: 22px 24px;
}
.slip .decision { font-size: 1.9rem; font-weight: 700; color: var(--band); margin: 0; }
.slip .band { color: #3A4356; margin: 4px 0 18px 0; }

.scale { position: relative; height: 12px; display: flex; border-radius: 3px; overflow: hidden; }
.scale div { height: 100%; }
.marker-row { position: relative; height: 22px; }
.marker {
    position: absolute; top: 0; transform: translateX(-50%);
    width: 0; height: 0;
    border-left: 7px solid transparent; border-right: 7px solid transparent;
    border-bottom: 10px solid #1F2A44;
}
.scale-labels { display: flex; justify-content: space-between; color: #5B6475; font-size: 0.82rem; }

.facts { width: 100%; border-collapse: collapse; margin-top: 18px; font-variant-numeric: tabular-nums; }
.facts td { padding: 7px 0; border-bottom: 1px solid #EEF1F5; color: #3A4356; }
.facts td:last-child { text-align: right; font-weight: 600; color: #1F2A44; }
.flag { color: #B23A3A; font-weight: 600; }

.empty {
    background: #FFFFFF; border: 1px dashed #C4CBD7; border-radius: 6px;
    padding: 28px 24px; color: #5B6475;
}
.note { color: #6B7385; font-size: 0.82rem; margin-top: 14px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ── DROPDOWN OPTIONS ─────────────────────────
emp_options = ["Less than 1 year", "1 year"] + [f"{i} years" for i in range(2, 10)] + ["10+ years"]

home_options = {"Rent": "RENT", "Own": "OWN", "Mortgage": "MORTGAGE"}

verify_options = {"Not verified": "Not Verified",
                  "Income source verified": "Source Verified",
                  "Income verified": "Verified"}

# purposes and states come straight from the training data
purposes = list(encoders['purpose'].classes_)
purpose_labels = {p.replace('_', ' ').capitalize(): p for p in purposes}
states = sorted(encoders['addr_state'].classes_)


# ── HEADER ───────────────────────────────────
st.title("Credit Risk Review")
st.markdown('<p class="subtitle">Enter the application details to get a risk band and a recommended decision.</p>',
            unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")


# ── APPLICATION FORM ─────────────────────────
with left:
    with st.form("application"):

        st.markdown('<div class="section">Applicant</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        annual_inc = a1.number_input("Annual income ($)", min_value=10000, max_value=1000000,
                                     value=65000, step=1000)
        emp_choice = a2.selectbox("Employment length", emp_options, index=5)
        home_choice = a3.selectbox("Home ownership", list(home_options.keys()))

        b1, b2 = st.columns(2)
        verify_choice = b1.selectbox("Income verification", list(verify_options.keys()))
        addr_state = b2.selectbox("State", states, index=states.index('MA') if 'MA' in states else 0)

        st.markdown('<div class="section">Loan request</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        loan_amnt = c1.number_input("Loan amount ($)", min_value=1000, max_value=40000,
                                    value=12000, step=500)
        term = c2.selectbox("Term", [36, 60], format_func=lambda x: f"{x} months")
        purpose_choice = c3.selectbox("Purpose", list(purpose_labels.keys()),
                                      index=list(purpose_labels.values()).index('debt_consolidation'))

        d1, d2 = st.columns(2)
        grade = d1.selectbox("Assigned grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2,
                             help="Grade from the pricing system")
        int_rate = d2.number_input("Interest rate (%)", min_value=5.0, max_value=31.0,
                                   value=14.0, step=0.25, format="%.2f")

        st.markdown('<div class="section">Credit file</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        fico_range_low = e1.number_input("FICO score", min_value=660, max_value=850, value=690, step=5,
                                         help="Model was trained on scores of 660 and above")
        dti = e2.number_input("Debt-to-income (%)", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
        revol_util = e3.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0,
                                     value=45.0, step=1.0)

        f1, f2, f3 = st.columns(3)
        revol_bal = f1.number_input("Revolving balance ($)", min_value=0, max_value=500000,
                                    value=12000, step=500)
        open_acc = f2.number_input("Open accounts", min_value=0, max_value=60, value=10, step=1)
        total_acc = f3.number_input("Total accounts", min_value=1, max_value=150, value=24, step=1)

        g1, g2, g3 = st.columns(3)
        delinq_2yrs = g1.number_input("Delinquencies, last 2 years", min_value=0, max_value=20, value=0, step=1)
        pub_rec = g2.number_input("Public records", min_value=0, max_value=10, value=0, step=1)
        months_delinq = g3.number_input("Months since last delinquency", min_value=0, max_value=200,
                                        value=0, step=1, help="Leave at 0 if there has never been one")

        submitted = st.form_submit_button("Run risk review", type="primary", use_container_width=True)


# ── DECISION PANEL ───────────────────────────
with right:
    if not submitted:
        st.markdown('<div class="empty">Fill in the application and select <b>Run risk review</b>. '
                    'The decision and the numbers behind it will show here.</div>',
                    unsafe_allow_html=True)

    elif open_acc > total_acc:
        st.error("Open accounts can't be more than total accounts. Check the credit file and run the review again.")

    else:
        # monthly payment using the standard loan formula
        monthly_rate = int_rate / 100 / 12
        installment = loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** -term)
        payment_to_income = installment / (annual_inc / 12)

        # 0 means no delinquency on record, same 999 code used in training
        if months_delinq == 0:
            mths_since_last_delinq = 999
        else:
            mths_since_last_delinq = months_delinq

        input_data = pd.DataFrame([{
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'grade': encoders['grade'].transform([grade])[0],
            'emp_length': emp_options.index(emp_choice),
            'home_ownership': encoders['home_ownership'].transform([home_options[home_choice]])[0],
            'annual_inc': annual_inc,
            'verification_status': encoders['verification_status'].transform([verify_options[verify_choice]])[0],
            'purpose': encoders['purpose'].transform([purpose_labels[purpose_choice]])[0],
            'addr_state': encoders['addr_state'].transform([addr_state])[0],
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'fico_range_low': fico_range_low,
            'mths_since_last_delinq': mths_since_last_delinq,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'payment_to_income': payment_to_income,
            'high_dti_flag': int(dti > 35)
        }])

        input_data = input_data[columns]
        prob = float(model.predict_proba(input_data)[0][1])

        # bands set by rank in notebook 05: safest 50% / next 30% / riskiest 20%
        low_cut = cutoffs['low_cut']
        high_cut = cutoffs['high_cut']

        if prob < low_cut:
            decision, band, color = "Approve", "Low", "#2E7D5B"
        elif prob < high_cut:
            decision, band, color = "Refer for review", "Medium", "#B7791F"
        else:
            decision, band, color = "Decline", "High", "#B23A3A"

        # things an underwriter would want to see next to the decision
        dti_text = f"{dti:.1f}%"
        if dti > 35:
            dti_text = f'<span class="flag">{dti:.1f}% (above 35%)</span>'

        util_text = f"{revol_util:.0f}%"
        if revol_util > 80:
            util_text = f'<span class="flag">{revol_util:.0f}% (above 80%)</span>'

        low_w = low_cut * 100
        mid_w = (high_cut - low_cut) * 100
        high_w = 100 - low_w - mid_w
        marker = min(max(prob, 0.01), 0.99) * 100

        st.markdown(f"""
<div class="slip" style="--band: {color};">
  <p class="decision">{decision}</p>
  <p class="band">{band} risk band, model score {prob:.1%}</p>

  <div class="scale">
    <div style="width:{low_w}%; background:#2E7D5B;"></div>
    <div style="width:{mid_w}%; background:#B7791F;"></div>
    <div style="width:{high_w}%; background:#B23A3A;"></div>
  </div>
  <div class="marker-row"><div class="marker" style="left:{marker}%;"></div></div>
  <div class="scale-labels"><span>Approve</span><span>Review</span><span>Decline</span></div>

  <table class="facts">
    <tr><td>Monthly payment</td><td>${installment:,.2f}</td></tr>
    <tr><td>Payment as share of monthly income</td><td>{payment_to_income:.1%}</td></tr>
    <tr><td>Debt-to-income</td><td>{dti_text}</td></tr>
    <tr><td>Revolving utilization</td><td>{util_text}</td></tr>
    <tr><td>FICO score</td><td>{fico_range_low}</td></tr>
    <tr><td>Grade and rate</td><td>{grade}, {int_rate:.2f}%</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

        st.markdown(f'<p class="note">The score ranks applicants by relative risk. It is not a literal '
                    f'probability of default. Bands come from the training data: the safest 50% of '
                    f'applicants are approved, the next 30% referred, and the riskiest 20% declined. '
                    f'Model: XGBoost trained on LendingClub loans, 2007 to 2018.</p>',
                    unsafe_allow_html=True)
