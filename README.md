# 🚗 Predictive Modeling for Used Car Pricing  
*A Machine Learning Project*

---

## 📌 Project Overview

This project applies machine learning to help used car dealerships optimize pricing strategies.  
Using a comprehensive dataset of historical vehicle transactions, the objective is to develop predictive models that estimate the fair market value of a used car based on key features, including:

- Make, model, and year  
- Condition and odometer reading  
- Manheim Market Report (MMR) value  
- Final selling price

By leveraging supervised regression techniques, the project aims to identify the most influential pricing factors, evaluate model accuracy, and assess generalizability across different vehicle categories. The ultimate goal is to provide actionable insights to support data-driven pricing and inventory decisions.

---

## 🎯 Expected Outcomes

- **Identify the most influential features** — such as MMR, condition, and odometer — that drive used vehicle pricing decisions.  
- **Build and evaluate high-performing ML models** (e.g., XGBoost, LightGBM) to accurately predict vehicle selling prices with minimal error.  
- **Assess model fairness and generalization** across key subgroups (e.g., SUVs vs. others, Japanese vs. non-Japanese brands), ensuring pricing models are reliable and unbiased.  
- **Provide data-driven recommendations** to support smarter pricing, inventory sourcing, and model deployment in real-world business settings.

## 📊 Dataset

- **Source:** [Kaggle - Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
- **Size:** ~450,000 records
- **Key Features:**
  - `make`, `model`, `year`, `condition`, `odometer`, `color`, `brand`, `category`
  - `mmr` (Manheim Market Report value)
  - `sellingprice` (target variable)
 
- 📁 **Final Dataset Included**

- [`data/final_selected_features_car_prices.csv`](./data/final_selected_features_car_prices.csv)  
  This file contains the cleaned and feature-selected dataset used for model training. Intermediate feature selection datasets are documented in the notebook but not included to reduce redundancy.

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

📁 `[report](./report/report.md)

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

---

📘 **View full notebook links and instructions here:** [notebook_link.md](notebook_link.md)
