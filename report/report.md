# 🚗 Predicting Used Car Prices with Machine Learning

This project was developed for **ISOM 835: Predictive Analytics**. It explores how machine learning can be used to predict the selling price of used cars based on real-world vehicle data.

---

## 🎯 Project Objectives

- Identify which features most strongly influence used car prices.
- Build and evaluate ML regression models to predict vehicle prices.
- Analyze feature importance and residuals to assess accuracy and generalizability.
- Provide insights that could support smarter pricing for dealerships and resale platforms.

---

## 📊 Dataset Description

- **Source:** [Kaggle - Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
- **Size:** ~451,000 rows after cleaning
- **Key Features:**
  - `make`, `model`, `year`, `condition`, `odometer`
  - `mmr` (Manheim Market Report value)
  - `sellingprice` (target variable)
- The dataset offers rich, business-relevant data for training accurate, real-world regression models.

---

## 🧠 Models and Tools

- **Models Used:** Linear Regression, Random Forest, Gradient Boosting, HistGradientBoosting, LightGBM, XGBoost
- **Tools:** Google Colab, pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, lightgbm, joblib

---

## ⚙️ Model Training & Evaluation

After data cleaning and preprocessing, several ML models were trained and compared. The best performance was achieved by the **XGBoost** model:

| Metric       | Value     |
|--------------|-----------|
| R² (Test)    | 0.9762    |
| RMSE (Test)  | 0.1542    |
| MAE (Test)   | 0.1014    |

Other models like LightGBM and HistGradientBoosting also performed similarly well.

---

## 🔍 Feature Importance

- **MMR** was the most important feature across all models, accounting for over **99%** of the predictive power.
- **Odometer** and **Condition** had some impact, but all other features (e.g., year, brand, color) contributed very little.

> ⚠️ Over-reliance on `mmr` could pose risks if it’s missing or inaccurate during deployment.

---

## 📈 Model Insights & Residuals

- The residuals were centered around zero, with no major bias.
- The models showed good generalization across categories (e.g., SUV vs. Other, Japanese vs. Other brands).
- Slightly higher error was observed for high-priced or rare vehicles.

---

## 📌 Business Questions Answered

1. **Which factors most influence used vehicle price?**  
   → `mmr` is by far the strongest predictor. Condition and odometer add minor predictive value.

2. **How accurately can prices be predicted?**  
   → Ensemble models like XGBoost and LightGBM achieve high accuracy (R² > 0.976), with low RMSE and MAE.

3. **Are there subgroups where prediction varies?**  
   → Slightly higher error for SUVs and non-Japanese brands, but overall performance is consistent across groups.

---

## 📁 GitHub & Notebooks

- GitHub Repo: [🔗 Your Repository Link Here]
- Google Colab Notebook: [🔗 Your Colab Link Here]

---

## ✅ Conclusion

This project demonstrates the power of machine learning in predicting used vehicle prices with high accuracy and interpretability. Key business insights include the dominance of market benchmark features (like `mmr`) and the ability of ensemble models to generalize well across vehicle types.

---

## 🙋‍♂️ Author

**[Your Name]**  
MSBA Candidate, Suffolk University  
Date: May 2025
