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




**Used Car Price Prediction Report**

---

### 📈 Business Question 1: Which factors most significantly influence the selling price of used vehicles?

To determine the most important predictors of used vehicle prices, we examined feature importances across top-performing models. All models unanimously identify **MMR (Manheim Market Report)** as the dominant feature.

#### • Key Feature Importance Findings:

* **MMR Value** accounts for over **99%** of importance in models like XGBoost and Gradient Boosting.

  * Strong correlation between MMR and actual selling price.
  * Confirms MMR is a reliable pricing anchor.

* **Secondary Features** (all <1% importance):

  * **Condition**: Slight influence; better condition has minor positive impact.
  * **Odometer**: Lower mileage correlates with slightly higher price.
  * **Year**: Newer cars tend to be priced higher, marginally.
  * **Other features** (e.g., color, brand, body type) have **negligible predictive power** (<0.01%).

#### • Business Implication:

> Use MMR as the core reference for setting vehicle prices. Other features can be used for segmentation, marketing, and inventory decisions.

---

### 🔢 Business Question 2: How accurately can the selling price of a vehicle be predicted based on its features?

Model performance was evaluated using R^2, RMSE, MAE, and overfit gap. Top ensemble-based models showed high accuracy and minimal overfitting.

#### • Top 3 Performing Models:

| Model Type        | R^2 (Test) | RMSE   | MAE    | Overfit Gap |
| ----------------- | ---------- | ------ | ------ | ----------- |
| Gradient Boosting | 0.9764     | 0.1534 | 0.1008 | 0.00089     |
| LightGBM          | 0.9763     | 0.1536 | 0.1008 | 0.00079     |
| HistGBM           | 0.9763     | 0.1538 | 0.1012 | 0.00015     |

* **Linear Regression** underperformed with R^2 = 0.9746 and higher error metrics.
* Residual plots and distributions show:

  * Errors are symmetrically centered around zero.
  * No major heteroscedasticity.
  * Residuals are not normally distributed (Shapiro-Wilk p < 0.001).

#### • Business Implication:

> Ensemble models generalize well and offer strong predictive performance. The Gradient Boosting Regressor is the most suitable model for production deployment.

---

### 🔎 Business Question 3: How well does the predictive model generalize across different vehicle categories or brands?

Performance analysis across vehicle **categories (SUV vs Other)** and **brands (Japanese vs Other)** reveals strong generalization with minor gaps:

#### • By Vehicle Type:

| Category            | MAE    | R^2    | Count  |
| ------------------- | ------ | ------ | ------ |
| SUVs and Crossovers | 0.1042 | 0.9782 | 30,259 |
| Other Vehicles      | 0.0990 | 0.9750 | 59,990 |

* Slightly **higher error variance** for SUVs.
* Model slightly better on non-SUVs due to **more stable pricing**.

#### • By Brand:

| Brand          | MAE    | R^2    | Count  |
| -------------- | ------ | ------ | ------ |
| Japanese Brand | 0.0958 | 0.9718 | 29,836 |
| Other          | 0.1032 | 0.9778 | 60,413 |

* **Tighter residual distribution** for Japanese brands.
* Indicates **more standardized resale behavior**.

#### • Combined Group Summary:

| Group             | MAE    | R^2    | Count  |
| ----------------- | ------ | ------ | ------ |
| Japanese Brand    | 0.0958 | 0.9718 | 29,836 |
| Other Brand       | 0.1032 | 0.9778 | 60,413 |
| SUVs & Crossovers | 0.1042 | 0.9782 | 30,259 |
| Other Vehicles    | 0.0990 | 0.9750 | 59,990 |

#### • Error Outliers:

* Top 10 prediction errors came from the "Other" category.
* Likely due to:

  * Rare models or configurations.
  * Market price volatility.
  * Potential data quality issues.

#### • Supporting Visuals:

* Residual vs. Predicted scatterplots
* Residual KDE and boxplots by category and brand
* Performance bar charts
* Table of top 10 worst residuals

#### • Business Implication:

> The model is fair and effective across subgroups. Additional adjustments may be considered for rare vehicle types or volatile segments.

---

### 🔮 Conclusion:

* **MMR** is the core driver of used vehicle prices.
* **Gradient Boosting** model offers highest accuracy and generalizability.
* Model generalizes well across brands and vehicle types, with some variance in SUVs and non-standard brands.
* The system is **robust, fair, and ready** for production use with minor caution for outlier vehicles.

---

