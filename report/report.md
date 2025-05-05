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

## 📈 Best Model Performance Summary

The Gradient Boosting Regressor (GBR) achieved the best predictive performance among all models evaluated. It balanced high accuracy with low overfitting, making it the most reliable model for this use case.

| Model                     | Test R² | Test RMSE | Test MAE | Overfit Gap |
|--------------------------|---------|-----------|----------|--------------|
| Gradient Boosting (GBR)  | **0.9764** | 0.1534    | 0.1008   | 0.0009       |

- **Test R²** of 0.9764 indicates excellent fit and strong predictive power.
- **Overfit Gap** (Train R² - Test R²) is only 0.0009, suggesting minimal overfitting.
- **MAE** and **RMSE** values are low, showing accurate average predictions and low error dispersion.

✅ **Conclusion:**  
Gradient Boosting not only outperforms other models in accuracy, but also demonstrates excellent generalizability—making it well-suited for deployment in pricing applications.

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
