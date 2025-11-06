# Loan Default Risk Prediction

A machine learning project to predict the likelihood that a loan will be fully paid versus charged off.  
The work is implemented in a Jupyter Notebook and compares multiple classifiers using standard evaluation metrics.

## Project Structure
```
.
├─ LoanPrediction-Classification.ipynb   # Main notebook (EDA → cleaning → feature engineering → modeling → evaluation)
├─ LoansTrainingSet.csv                  # Training dataset
└─ .ipynb_checkpoints/                   # Jupyter checkpoints (ignored in Git)
```

## Problem Statement
Given historical customer and loan attributes, predict whether a customer is a **good borrower** (fully paid) or a **defaulter** (charged off). The goal is to help credit officers screen applications and ask **targeted follow‑up questions** during the application process.

## Data Dictionary (short)
- **Loan ID**: Unique loan identifier  
- **Customer ID**: Unique borrower identifier (a customer may have multiple loans)  
- **Loan Status**: Target label; `'Fully Paid'` or `'Charged Off'`  
- **Current Loan Amount**: Current principal balance  
- **Term**: Loan term (short / long)  
- **Credit Score**: 0–800 credit risk score  
- **Years in current job**: Tenure in current role (years)  
- **Home Ownership**: Rent / Home Mortgage / Own  


## Workflow
1. **EDA** – basic checks, distributions, outliers.  
2. **Cleaning**
   - Handle missing values; fix formatting issues; drop identifier columns (`Loan ID`, `Customer ID`).
   - Filter obvious outliers in `Current Credit Balance` and similar columns.
   - Map target to binary: `{'Fully Paid': 1, 'Charged Off': 0}`.
3. **Feature Engineering**
   - Example transformations (illustrative):
     - `Credit Score²`, `Years of Credit History²`, `Bankruptcies²`
     - **DTI Ratio**: `Monthly Debt / (Annual Income / 12 + 1)`
   - One‑hot encode categorical fields.
4. **Modeling**
   - Compared: **Logistic Regression**, **RandomForestClassifier**, **GaussianNB**, **BernoulliNB**.
   - Split: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
5. **Evaluation**
   - **Accuracy**, **Precision**, **Recall**, **F1‑score**, **ROC‑AUC**.
   - Results aggregated in a DataFrame for side‑by‑side comparison.

## Reproducibility
- Python ≥ 3.10  
- Key packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`

### Quickstart
```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt  # or install the packages listed above
jupyter notebook
# Open LoanPrediction-Classification.ipynb and run the cells
```

> **Note:** The dataset may contain personally identifiable information in real scenarios. Make sure to comply with your organization’s data handling policies.

## Results
The notebook prints a table like:
Model  Accuracy  Precision    Recall  F1-score   ROC-AUC
0                  Linear  0.587863   0.744204  0.607902  0.669183  0.605830
1  RandomForestClassifier  0.807733   0.808938  0.942123  0.870466  0.829558
2              GaussianNB  0.618243   0.754883  0.656399  0.702204  0.629589
3             BernoulliNB  0.691939   0.707360  0.939352  0.807014  0.622077

## License
This project is licensed under the **MIT License** (see `LICENSE`).

## Acknowledgements
- This project was prepared for a credit‑risk screening use‑case.  
- The data dictionary file provides the full description of the variables.
