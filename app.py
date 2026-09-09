import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#  Custom transformer classes (must match the ones used to build the pipeline)

class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Converts biologically invalid zeros to NaN for specified columns."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = X[self.cols].replace(0, np.nan)
        return X


class SkinThicknessCapper(BaseEstimator, TransformerMixin):
    """Caps SkinThickness at a percentile threshold learned from training data."""
    def __init__(self, col='SkinThickness', quantile=0.99):
        self.col = col
        self.quantile = quantile

    def fit(self, X, y=None):
        self.upper_limit_ = X[self.col].quantile(self.quantile)
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X[self.col] > self.upper_limit_, self.col] = self.upper_limit_
        return X


#  Load the trained pipeline 
pipeline = joblib.load("diabetes_pipeline.pkl")

#  App layout 
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This tool estimates diabetes risk based on routine health measurements. "
    "Fill in the values below and click **Predict**."
)
st.caption("⚠️ This is a screening tool for a learning project, not a medical diagnosis.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    # Pipeline expects a DataFrame with the same column names used during training
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High diabetes risk** — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low diabetes risk** — estimated probability: {probability:.1%}")

    st.progress(probability)

    with st.expander("What influenced this prediction?"):
        model = pipeline.named_steps['model']
        importances = model.feature_importances_
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            st.write(f"- **{name}**: {score:.1%} importance")

st.divider()
st.caption("Built with scikit-learn + Streamlit · Trained on the Pima Indians Diabetes Dataset")