# 🚗 Predicting Used Car Prices with Machine Learning

This project was completed for **ISOM 835: Predictive Analytics**. It aims to build a robust machine learning model to accurately estimate the selling price of used vehicles using historical transaction data and vehicle features.

---

## 🎯 Project Objectives

- Identify which features most significantly affect used car prices.
- Build and evaluate ML regression models to predict vehicle prices.
- Analyze feature importance and residuals to improve model transparency.
- Provide actionable insights to support smarter pricing and inventory strategies for car dealerships.

---

## 📊 Dataset

- **Source:** [Kaggle - Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
- **Size:** ~450,000 records
- **Key Features:**
  - `make`, `model`, `year`, `condition`, `odometer`, `color`, `brand`, `category`
  - `mmr` (Manheim Market Report value)
  - `sellingprice` (target variable)

---

## 🛠️ Tools & Libraries

- **Google Colab (Python)**
- Libraries used:
  - `pandas`, `numpy`, `scikit-learn`
  - `xgboost`, `lightgbm`, `matplotlib`, `seaborn`, `joblib`

---

## 🔗 Google Colab Notebook

Open this notebook in Google Colab to view code, run the analysis, or reproduce the results:

➡️ [Google Colab: Used Car Price Prediction](https://drive.google.com/file/d/1wIt18lFApKYCF4RjMVaVUPg5u4kyeB3i/view?usp=sharing)

---

## ✅ How to Run This Project

1. Click the Google Colab link above.
2. Select `File > Save a Copy in Drive` to edit your own version.
3. Run the notebook step-by-step:
   - Load and preprocess the dataset
   - Train and evaluate multiple ML models (e.g., LightGBM, XGBoost, Random Forest)
   - Visualize feature importance and residual patterns
   - Generate summary insights

---

## 📄 Final Report

You can include a final report PDF or markdown summary here once it’s ready:

📁 `./reports/Final_Project_Report.pdf` *(Upload via GitHub)*

---

## 📁 Folder Structure

---

## 💡 Key Takeaways

- `mmr` was the most important feature across all models.
- Ensemble methods like **HistGB**, **Gradient Boosting**, **XGBoost** and **LightGBM** delivered the best performance (Test R² ≈ 0.976).
- The project shows how predictive analytics can improve pricing accuracy and inventory management in the used car market.

---

## 🙋 Author

**Willzh2025**  
MSBA Candidate, Suffolk University  
[LinkedIn](https://linkedin.com/in/yourname) *(Optional)*
