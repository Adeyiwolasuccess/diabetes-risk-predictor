# 🩺 Diabetes Risk Predictor

A machine learning web app that estimates diabetes risk based on routine health measurements, built end-to-end from raw data to a deployed Streamlit app.

**🔗 Live app:** _[https://diabetes-risk-predictor-ap.streamlit.app/]

---

## Overview

This project predicts whether a patient is likely diabetic using the Pima Indians Diabetes Dataset. It covers the full ML workflow: data cleaning, leakage-safe preprocessing, model comparison, hyperparameter tuning, and deployment as an interactive app.

## Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Rows:** 768 patients
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** `Outcome` (0 = non-diabetic, 1 = diabetic)

## Process

1. **Data cleaning** — identified biologically invalid zeros (e.g. Glucose/BMI = 0) as disguised missing values; capped a physiologically implausible outlier in Skin Thickness.
2. **Leakage-safe pipeline** — built custom scikit-learn transformers (`ZeroToNaN`, `SkinThicknessCapper`) combined with `SimpleImputer` inside a `Pipeline`, so all preprocessing is fit only on training data.
3. **Model comparison** — evaluated Logistic Regression (default and class-weighted) against Random Forest, using precision/recall/F1 rather than accuracy alone, due to class imbalance (~65/35).
4. **Tuning** — diagnosed overfitting in the default Random Forest (train acc. 1.00 vs. test 0.86), then regularized with `max_depth` and `min_samples_leaf`; validated further with `GridSearchCV`.
5. **Interpretability** — reviewed feature importances (Insulin, Glucose, and Skin Thickness ranked highest).
6. **Deployment** — packaged the final pipeline with `joblib` and built an interactive Streamlit app for real-time predictions.

## Results

Final model: **Random Forest** (`max_depth=5`, `min_samples_leaf=5`, `class_weight='balanced'`)

| Metric (Diabetic class) | Score |
|---|---|
| Precision | 0.80 |
| Recall | 0.87 |
| F1-score | 0.83 |
| Overall Accuracy | 0.88 |

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (pipelines, custom transformers, Random Forest, Logistic Regression)
- Streamlit (app/UI)
- joblib (model serialization)

## Project Structure

```
diabetes_app/
├── app.py                    # Streamlit app
├── diabetes_pipeline.pkl     # Trained, saved pipeline (preprocessing + model)
└── requirements.txt          # Dependencies
```

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This is a learning project and a screening-style demo — **not** a medical diagnostic tool. Predictions should not be used for real clinical decisions.

## Author

Built by Success as a first end-to-end machine learning project — covering data cleaning, leakage-safe pipelines, model evaluation, and deployment.