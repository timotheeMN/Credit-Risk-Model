# Credit Risk Modeling & Prediction App (Python, ML, Streamlit)

This repository presents an **end-to-end credit risk modeling project**, from exploratory data analysis to machine learning modeling and deployment via a **Streamlit web application**.

The goal is to predict whether a loan applicant is likely to **default (high risk)** or **repay (low risk)** based on demographic, financial, and loan-related features.


##  Project Overview

**Key objectives**
- Explore and understand credit risk data (EDA)
- Build and evaluate machine learning models
- Select and tune a robust classifier (XGBoost)
- Deploy the final model in an interactive Streamlit application

**Target variable**
- `Risk` → Good vs Bad credit


##  Dataset

- **German Credit Risk — With Target**
- Source: Kaggle  
- Link: see `data/data.txt`

⚠️ The dataset CSV is **not included** in the repository.  
Please download it from Kaggle and place it in the `data/` folder.


##  Methodology

### 1. Exploratory Data Analysis (EDA)
Performed in `notebooks/analysis_model.ipynb`:
- Numerical distributions & outlier detection
- Categorical feature analysis
- Correlation analysis
- Group-based statistics
- Multivariate visualizations

### 2. Feature Engineering & Preprocessing
- Selection of business-relevant features
- Encoding of categorical variables
- Train / test split with stratification
- Prevention of data leakage

### 3. Modeling
- Baseline and tree-based models
- Hyperparameter tuning using `GridSearchCV`
- Final model: **XGBoost Classifier**
- Evaluation on unseen test data

### 4. Deployment
- Trained model and encoders saved with `joblib`
- Interactive **Streamlit app** for real-time prediction

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
streamlit run notebooks/app.py
