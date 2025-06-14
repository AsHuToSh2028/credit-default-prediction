# Credit Default Risk Prediction

A machine learning project to predict credit card defaulters based on customer behavior. Developed as part of a FinClub Summer Project at IIT Roorkee (June 2025).

## 🔍 Objective

To identify high-risk customers by modeling payment delays, credit utilization, spending volatility, and demographic signals.

## 🧠 Key Highlights

- 📊 Feature engineering: Delay × Utilization, repayment volatility, usage clusters
- 🔎 Customer segmentation: KMeans on repayment and credit behavior
- 🌲 Modeling: LightGBM + hyperparameter tuning using Optuna
- 🎯 Thresholding: F2-score optimization for recall-focused predictions
- 🧮 Explainability: SHAP values + surrogate tree

## 🛠️ Tools & Libraries

- Python (pandas, NumPy)
- scikit-learn, LightGBM, XGBoost
- Optuna for tuning
- SHAP for interpretability
- Matplotlib, Seaborn

---
