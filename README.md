# Credit Risk Prediction System 

A machine learning system that predicts loan default risk using real LendingClub loan data (2007–2018).

## What This Project Does
A loan application comes in and has already been given a grade and interest rate by the lender's pricing system. This project acts as a second check: it looks at the applicant's full profile and predicts how risky they are compared to other borrowers. The output is a risk band (Low, Medium or High) with a recommended action (Approve, Refer for review or Decline).

## App
Built with Streamlit. The app works like an internal credit review screen: the reviewer enters the applicant, loan and credit file details, and gets a decision, a risk score and a short summary (monthly payment, payment as a share of income, DTI and utilization, with flags for high values).

The score ranks applicants by relative risk. It is not a literal probability of default, because the model was trained with extra weight on defaulters.

## Results
| Model | Accuracy | AUC-ROC | Recall (Defaulters) |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.704 | 7% |
| Random Forest | 63.9% | 0.712 | 68% |
| XGBoost | 65.2% | 0.724 | 68% |

**Risk Segmentation (policy: approve safest 50%, review next 30%, decline riskiest 20%):**
- Low Risk → 9.8% default rate → Recommend Approve
- Medium Risk → 23.1% default rate → Recommend Review
- High Risk → 40.6% default rate → Recommend Decline (captures ~40% of all defaulters)

The cutoffs are set from the training data, not the test data. Where to draw them is a business decision, not a model result.

**Top features:** loan grade was the strongest predictor by far (~74% of the model's importance), followed by term and interest rate. Grade is LendingClub's own risk rating, so part of what the model learns is already built into the grade.

## Dataset
- Source: LendingClub via Kaggle (2007–2018)
- Size: 2.26 million loan records loaded; 1.35 million with a final outcome used for modeling
- Features: 21 columns selected from 150+ using financial logic, only information available at application time (no post-loan payment data)
- Target: default (1 = Charged Off, 0 = Fully Paid). "Current" loans were removed because their outcome is unknown
- Default rate: 20% — 1 in 5 borrowers defaulted

## Project Structure
```
credit-risk-model/
├── 01_data_loading.ipynb        # Chunked loading of the 1.6GB file
├── 02_data_cleaning.ipynb       # Missing values, target variable, type conversion
├── 03_eda_visualization.ipynb   # Charts and EDA findings
├── 04_feature_engineering.ipynb # Encoding, new features, train-test split
├── 05_model_building.ipynb      # 3 models, evaluation, risk segmentation
├── outputs/
│   ├── xgb_model.pkl            # Saved XGBoost model
│   ├── model_columns.pkl        # Feature order the model expects
│   ├── encoders.pkl             # Category encoders from training, reused by the app
│   ├── risk_cutoffs.pkl         # Risk band cutoffs
│   └── *.png                    # Charts
├── app.py                       # Streamlit app
├── .gitignore
└── README.md
```

## How to Run

Install what you need:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
```

1. Download `accepted_2007_to_2018Q4.csv` from the LendingClub dataset on Kaggle and put it in a `data` folder inside the project.
2. Update the `os.chdir(...)` path at the top of each notebook to your project folder.
3. Run the notebooks in order from 01 to 05.
4. Launch the app:
```bash
streamlit run app.py
```

## Revisions
After reviewing the first version, I found and fixed a few issues:
- The app encoded inputs differently from training, so predictions were wrong.
  Fixed by saving the training encoders and reusing them in the app.
- payment_to_income used the loan amount instead of the monthly payment.
- The original risk cutoffs placed 68% of applicants in High Risk.
  Bands are now set by rank: safest 50% approve, next 30% review, riskiest 20% decline.

## Limitations
- The data only includes loans LendingClub approved, so the model has never seen rejected applicants.
- Train and test data were split randomly across 2007–2018, not by time.
- Grade and interest rate are inputs, so the model works as a check after pricing, not as a pricing tool.

## Future Improvements
- Train a version without grade and interest rate, so the model can recommend a grade and rate from the applicant's own details (risk-based pricing).
- Build a portfolio monitoring version for existing loans that uses payment behavior (late payments, recent FICO changes).
- Test on later years after training on earlier ones, to see how the model holds up over time.

## Tools

Python — Pandas — NumPy — Scikit-learn — XGBoost — Matplotlib — Seaborn — Streamlit
