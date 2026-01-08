import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# Folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

# Load model + encoders (filenames must match what you actually have)
model = joblib.load(MODELS_DIR / "XGBoost_credit_model.pkl")

encoders = {
    "Sex": joblib.load(MODELS_DIR / "Sex_encoder.pkl"),
    "Housing": joblib.load(MODELS_DIR / "Housing_encoder.pkl"),
    "Saving accounts": joblib.load(MODELS_DIR / "Saving accounts_encoder.pkl"),
    "Checking account": joblib.load(MODELS_DIR / "Checking account_encoder.pkl"),
}


st.title("Credit Risk Prediction App")
st.write("Enter the details of the applicant to predict credit risk.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", list(encoders["Sex"].classes_))
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", list(encoders["Housing"].classes_))
saving_accounts = st.selectbox("Saving accounts", list(encoders["Saving accounts"].classes_))
checking_account = st.selectbox("Checking account", list(encoders["Checking account"].classes_))
credit_amount = st.number_input("Credit Amount", min_value=0.0, value=1000.0)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# IMPORTANT: column names must match training exactly
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],  # <- use "Duration" if that's what you trained on
})

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The applicant is likely to repay the credit (Low Risk).")
    else:
        st.error("The applicant is likely to default on the credit (High Risk).")
