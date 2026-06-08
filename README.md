# Credit Risk Prediction System 🏦

A machine learning system that predicts loan default risk using 2.26 million real LendingClub loans. Built as part of MSIS-415 Advanced Coding for Analytics at UMass Boston.

## What This Project Does
A borrower walks into a bank and applies for a loan. This system analyzes their financial profile and predicts — in milliseconds — whether they are likely to default. The output is a risk category (Low, Medium, or High) with a recommended bank action (Approve, Review, or Decline).

## Live Demo
Built with Streamlit — run locally using the instructions below.

## Results
| Model | Accuracy | AUC-ROC | Recall (Defaulters) |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.694 | 5% |
| Random Forest | 63.8% | 0.712 | 68% |
| XGBoost ✅ Winner | 65.2% | 0.724 | 68% |

**Risk Segmentation:**
- Low Risk → 2.8% default rate → Recommend Approve
- Medium Risk → 8.2% default rate → Recommend Review  
- High Risk → 26.0% default rate → Recommend Decline

## Dataset
- Source: LendingClub via Kaggle (2007–2018)
- Size: 2.26 million real loan records
- Features: 21 selected from 150+ based on financial logic
- Target: default (1 = Charged Off, 0 = Fully Paid)
- Default rate: 20% — 1 in 5 borrowers defaulted

## Project Structure
credit-risk-model/
├── notebooks/
│   ├── 01_data_loading.ipynb       # Chunked loading of 1.6GB file
│   ├── 02_data_cleaning.ipynb      # Missing values, type conversion
│   ├── 03_eda_visualization.ipynb  # 5 charts, EDA findings
│   ├── 04_feature_engineering.ipynb # Encoding, new features, train-test split
│   └── 05_model_building.ipynb     # 3 models, evaluation, risk segmentation
├── outputs/
│   └── xgb_model.pkl               # Saved XGBoost model
├── app.py                          # Streamlit web application
├── .gitignore
└── README.md

---

## How to Run

Install what you need:
`ash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
`

Run notebooks in order from 01 to 05, then launch the app:
`ash
streamlit run app.py
`

---

## Tools

Python — Pandas — NumPy — Scikit-learn — XGBoost — Matplotlib — Seaborn — Streamlit
