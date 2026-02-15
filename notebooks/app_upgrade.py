import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

# Load model + encoders 
pipeline = joblib.load(MODELS_DIR / "Credit_risk_pipeline.pkl")
target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")

# Identify which label corresponds to "bad"
BAD_LABEL = int(np.where(target_encoder.classes_ == "bad")[0][0])

# Helper functions
def predict_pd(input_df: pd.DataFrame):
    proba = pipeline.predict_proba(input_df)[0]
    pd_bad = float(proba[BAD_LABEL])
    pred_encoded = int(np.argmax(proba))
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    return pred_label, pd_bad

def get_top_drivers(input_df: pd.DataFrame, top_n: int = 5):
    """
    Model explanation using feature importances (tree-based models).
    Works for RandomForest / ExtraTrees / XGBoost (if sklearn wrapper exposes feature_importances_).
    Returns a DataFrame with top features.
    """
    # Access preprocessor + classifier inside pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    # Transform input to numeric matrix
    X_trans = preprocessor.transform(input_df)

    # Feature names after preprocessing (numeric + onehot)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # fallback if old sklearn
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # Get feature importances
    if not hasattr(clf, "feature_importances_"):
        return None

    importances = clf.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    return df_imp

# UI
st.title("Credit Risk Prediction App")
st.write("Enter the details of the applicant to predict credit risk.")
st.caption("Predict default risk (PD) and explain key drivers using a trained ML pipeline.")


with st.sidebar:
    st.header("Applicant Information")

    age = st.slider("Age", min_value=18, max_value=80, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.slider("Job (0-3)", min_value=0, max_value=3, value=1)

    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
    checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])

    credit_amount = st.number_input("Credit Amount", min_value=0.0, value=1000.0, step=100.0)
    duration = st.slider("Duration (months)", min_value=1, max_value=72, value=12)

    threshold = st.slider("Decision threshold (PD Bad)", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
    show_drivers = st.checkbox("Show main risk drivers", value=True)

# Create input DataFrame
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Job": [job],
    "Housing": [housing],
    "Saving accounts": [saving_accounts],
    "Checking account": [checking_account],
    "Credit amount": [credit_amount],
    "Duration": [duration],
})

st.subheader("Prediction")

if st.button("Predict"):
    pred_label, pd_bad = predict_pd(input_df)

    # Decision based on threshold
    decision_bad = pd_bad >= threshold

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability of Default (PD)", f"{pd_bad*100:.1f}%")

    with col2:
        st.metric("Threshold", f"{threshold*100:.0f}%")

    st.progress(min(max(pd_bad, 0.0), 1.0))

    if decision_bad:
        st.error(f"High Risk (Bad) — PD {pd_bad*100:.1f}% ≥ threshold {threshold*100:.0f}%")
    else:
        st.success(f"Low Risk (Good) — PD {pd_bad*100:.1f}% < threshold {threshold*100:.0f}%")

    # Also show raw model label
    with st.expander("Model output details"):
        st.write(f"Model predicted label: **{pred_label}**")
        st.write("Note: final decision is based on the threshold policy above.")
         # Drivers (feature importance)
    if show_drivers:
        st.subheader("Main Risk Drivers (Model-level importance)")
        df_imp = get_top_drivers(input_df, top_n=8)
        if df_imp is None:
            st.info("This model does not expose feature_importances_.")
        else:
            st.dataframe(df_imp, use_container_width=True)
            st.caption("These are global importance scores from the trained model.")
else:
    st.info("Fill the sidebar and click **Predict** to generate a PD and decision.")
