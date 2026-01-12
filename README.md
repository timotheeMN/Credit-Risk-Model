# Credit Risk Modeling & Prediction App (Python, ML, Streamlit)

This repository presents an **end-to-end credit risk modeling project**, from exploratory data analysis to machine learning modeling and deployment via a **Streamlit web application**.

Goal: predict whether a loan applicant is likely to **default (Bad / High Risk)** or **repay (Good / Low Risk)** and provide a **probability of default (PD)** with basic model explanations.


##  Project Overview

**Key objectives**
- Perform Exploratory Data Analysis (EDA)
- Build and tune ML models for credit risk classification
- Use a production-style preprocessing pipeline (**OneHotEncoder + ColumnTransformer**)
- Evaluate with credit-risk relevant metrics (ROC-AUC, Recall on Bad loans)
- Deploy an interactive Streamlit demo (PD + threshold policy + drivers)

**Target variable**
- `Risk` → Good vs Bad credit


##  Dataset

- **German Credit Risk — With Target**
- Source: Kaggle  
- Link: see `data/data.txt`

⚠️ The dataset CSV is **not included** in the repository.  
Please download it from Kaggle and place it in the `data/` folder.


##  Methodology

### 1) EDA
Notebook: `notebooks/analysis_model.ipynb`  
Includes distributions, outliers, categorical analysis, correlations and multivariate plots.

### 2) Preprocessing (Production-ready)
Notebook: `notebooks/analysis_model_upgrade.ipynb`  
- Train/test split with stratification
- **ColumnTransformer**
  - numeric: median imputation
  - categorical: most-frequent imputation + **OneHotEncoder(handle_unknown="ignore")**
- Avoids artificial ordering from LabelEncoder on features

### 3) Modeling & Evaluation
- Tree-based models + hyperparameter tuning (GridSearchCV)
- Credit-risk evaluation:
  - Confusion matrix
  - **Recall on Bad loans**
  - Precision / F1
  - **ROC-AUC**
  - Threshold tuning using `predict_proba` (policy-based decision)

### 4) Deployment (Streamlit)
Streamlit app provides:
- PD (Probability of Default)
- Adjustable threshold for decision policy
- Basic explanation via global feature importance

---

##  Streamlit Application (Demo)

The application allows users to input applicant information and instantly receive a **credit risk prediction**.

### Run locally

```bash
# (Optional) Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run notebooks/app_upgrade.py
