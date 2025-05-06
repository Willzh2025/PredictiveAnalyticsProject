
# 🚗 Predictive Modeling for Used Car Pricing
> ——*A Machine Learning Project*


---

## 📌 Project Overview  

This project applies machine learning to help used car dealerships optimize pricing strategies.  
Using a comprehensive dataset of historical vehicle transactions, the objective is to develop predictive models that estimate the fair market value of a used car based on key features, including:

- Make, model, and year  
- Condition and odometer reading  
- Manheim Market Report (MMR) value  
- Final selling price  

By leveraging supervised regression techniques, the project aims to identify the most influential pricing factors, evaluate model accuracy, and assess generalizability across different vehicle categories. The ultimate goal is to provide actionable insights to support data-driven pricing and inventory decisions.

## 🎯 Expected Outcomes

- **Identify the most influential features**—such as MMR, condition, and odometer—that drive used vehicle pricing decisions.
- **Build and evaluate high-performing ML models** (e.g., XGBoost, LightGBM) to accurately predict vehicle selling prices with minimal error.
- **Assess model fairness and generalization** across key subgroups (e.g., SUVs vs. others, Japanese vs. non-Japanese brands), ensuring pricing models are reliable and unbiased.
- **Provide data-driven recommendations** to support smarter pricing, inventory sourcing, and model deployment in real-world business settings.

---

## Introduction & Dataset Description

In today’s used vehicle market, pricing strategies must be both dynamic and data-driven. Consumers seek fair prices while dealerships aim for profitability. To bridge this gap, predictive analytics can play a critical role in estimating fair market values of used cars. This project applies supervised machine learning techniques to a large-scale dataset of historical vehicle transactions to build a regression model that predicts used car selling prices based on a variety of features. The goal is to support used car dealerships in making informed pricing decisions and identifying patterns in consumer behavior.

### Dataset Overview

The dataset used in this project is the **"Vehicle Sales and Market Trends Dataset"**, originally published on [Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data). It contains detailed information on **558,837** used vehicle transactions, with **16 columns** describing each vehicle's specifications, condition, market benchmarks, and final selling price.

This real-world dataset provides a strong foundation for building predictive machine learning models to estimate used car prices — a task of direct business relevance to auto dealers, resale platforms, and pricing analysts.

### Dataset Summary


| Attribute                    | Value                    |
|-----------------------------|--------------------------|
| 📁 Dataset Name             | Vehicle Sales and Market Trends |
| 🔢 Number of Rows           | 558,837                  |
| 🔠 Number of Columns        | 16                       |
| 🧾 File Format              | CSV                      |
| 🔁 Update Frequency         | Periodically             |
| ✅ Data Integrity           | Reasonable quality, but further validation recommended |



### Dataset Features


The dataset includes a wide range of attributes that influence vehicle valuation:

- **Vehicle Details**: `make`, `model`, `year`, `trim`, `body type`, `transmission`, `VIN`
- **Condition & Mileage**: `condition`, `odometer`, `exterior_color`, `interior_color`
- **Transaction Info**: `sellingprice`, `date`, `state`, `seller`
- **Market Benchmark**: `mmr` (Manheim Market Report), which provides an industry pricing baseline


| **Column**       | **Description**                                                                 |
|------------------|----------------------------------------------------------------------------------|
| `year`           | Year of the vehicle                                                             |
| `make`           | Manufacturer brand (e.g., BMW, Toyota)                                          |
| `model`          | Specific model name                                                             |
| `trim`           | Trim level                                                                      |
| `body`           | Body type (e.g., SUV, Sedan)                                                    |
| `transmission`   | Transmission type (manual or automatic)                                         |
| `condition`      | Numeric condition rating (subjective or appraised)                              |
| `odometer`       | Mileage reading at time of sale                                                 |
| `color`          | Exterior color                                                                  |
| `interior`       | Interior color                                                                  |
| `seller`         | Selling party or dealer name                                                    |
| `mmr`            | Manheim Market Report value (benchmark vehicle price)                           |
| `sellingprice`   | Final sale price of the vehicle                                                 |
| `saledate`       | Timestamp of the sale transaction                                               |



### Summary Statistics

| Metric                     | Value     |
|---------------------------|-----------|
| Number of rows            | 558,837   |
| Number of columns         | 16        |
| Mean vehicle year         | 2010.0    |
| Mean odometer             | 68,320    |
| Mean condition            | 30.67     |
| MMR (avg)                 | $13,769   |
| Selling Price (avg)       | $13,611   |




### Sample Data Variety

Below are a few randomly sampled entries from different years, states, and sellers to illustrate the dataset’s diversity:

| year | make   | model     | state | condition | odometer | mmr   | sellingprice |
|------|--------|-----------|-------|-----------|----------|-------|---------------|
| 2013 | Ford   | Escape    | TX    | 4.0       | 17,098   | 16400 | 16700         |
| 2002 | Nissan | Sentra    | FL    | 36.0      | 85,538   | 1900  | 2500          |
| 2012 | Kia    | Optima    | PA    | 47.0      | 33,002   | 14500 | 15300         |
| 2011 | BMW    | 3 Series  | NJ    | 44.0      | 37,196   | 19650 | 18800         |



### Why This Dataset?

This dataset was selected for the following reasons:

- **Practical Business Relevance**: Used car pricing is a key challenge in auto sales and inventory management.
- **Rich Feature Set**: Includes both vehicle-level details and market-level signals (e.g., MMR).
- **Real-World Scale**: With over half a million records, it supports robust training, validation, and generalization.
- **Model Interpretability**: The presence of interpretable features like condition, mileage, and brand enables clear insights from ML models.



### Potential Use Cases

- ** Market Analysis**: Track pricing trends by year, region, or vehicle type.
- ** Predictive Modeling**: Build regression models to estimate car values.
- ** Business Strategy**: Support pricing, sourcing, and resale decisions with data-driven insights.
- ** Customer Profiling**: Understand patterns in buying behavior or vehicle depreciation.



### Project Relevance

This project applies supervised regression techniques to:

- Predict the **selling price** of a used vehicle based on its attributes
- Identify the **key drivers** of price variation (e.g., MMR, condition, mileage)
- Evaluate model performance across different **vehicle subgroups** (e.g., by category or brand)

The ultimate goal is to deliver actionable insights that support smarter, data-driven decisions for **used car dealerships** and **resale platforms**.


## Exploratory Data Analysis (EDA)

### Distribution of Categorical Features

The following chart presents the frequency distributions of key categorical features in the dataset:

![Distribution of Categorical Features](../visualizations/Categorical_Feature_Distributions.png)

Key Takeaways:

- **Make**: Ford, Chevrolet, and Nissan are the most frequently listed brands, representing a large portion of the dataset.
- **Body Type**: Sedans and SUVs dominate the vehicle body types, while coupes, convertibles, and wagons are less common.
- **Transmission**: Most vehicles use automatic transmission, suggesting potential simplification in modeling.
- **State**: Florida and California contribute the most records, indicating possible regional market trends.
- **Country of Origin**: Vehicles from the U.S. and Japan lead the dataset, with Germany a distant third.
- **Exterior & Interior Colors**: Black, white, and gray are dominant for both exterior and interior colors, suggesting consumer preferences.
- **Year Distribution**: The bulk of the vehicles were manufactured between 2010 and 2015, reflecting recency in resale activity.

These distributions provide important context for understanding the dataset composition and help inform preprocessing steps (e.g., rare category consolidation).

---

#### Top Vehicle Attributes (Categorical)

- **Top Makes**: Ford, Chevrolet, Nissan, Toyota, Honda.
- **Top Body Types**: Sedans and SUVs dominate.
- **Transmission Types**: Majority are automatic.
- **Top States**: FL, CA, PA, TX have the most listings.

![Top Vehicle Attributes (Categorical)](../visualizations/Top_10_Brands_Body_Types_Transmision_and_States.png)


---

#### Distribution by Category

- **Country of Origin**: U.S.-made vehicles dominate the dataset, followed by Japan and Germany.
- **Vehicle Type**: Sedans and SUVs account for the majority; pickup trucks and minivans are fewer.
- **Vehicle Color**: Black, white, silver, and gray are the most common exterior colors.
- **Interior Color**: Black and gray dominate interior designs, indicating market preference.

![Distribution by Category](../visualizations/country_distribution.png)
![Distribution by Category](../visualizations/bodygroup_distribution.png)
![Distribution by Category](../visualizations/exterior_color_distribution.png)
![Distribution by Category](../visualizations/interior_color_distribution.png)

---


#### Vehicle Transmission Breakdown

- Automatic transmissions (encoded as `1`) account for more than **96%** of all entries, reinforcing the dominance of automatics in the U.S. used vehicle market. Manual vehicles are rare in this dataset.

![Vehicle Transmission Breakdown](../visualizations/Vehicle_Transmission.png)

---

### Distribution of Key Numerical Features

#### Distribution of Selling Price Odometer and Odometer

- **Selling Price**: Right-skewed; most cars are priced between \$5,000 and \$20,000, with a long tail above \$100,000.
- **Odometer**: Positively skewed; most vehicles have mileage under 100,000 miles, but some outliers exceed 900,000.
- **Odometer (Market Benchmark Price)**: Also right-skewed, with values typically between \$5,000 and \$25,000.

![Distribution of Key Numerical Features](../visualizations/Distributions_of_sellingprice_odometer_and_mmr.png)

---

#### Log-Transformed Distribution of Selling Prices

- Selling price is heavily right-skewed in raw form, with a long tail toward high-end vehicles.  
- After log transformation, the distribution becomes much more symmetric and bell-shaped, which benefits linear model assumptions.

![Log-Transformed Distribution of Selling Prices](../visualizations/Log_Transformed_Distribution_of_Selling_Prices.png)


---

### Feature Relationships

---

#### Selling Price vs Numerical Features


##### Distribution of Selling Price vs Key Features

![Selling Price vs Odometer and MMR](../visualizations/Selling_Price_vs_Odometer_and_MMR.png)

**Key Insights:**

- **Odometer**: There is a clear negative relationship—vehicles with lower mileage tend to command higher prices. Outliers with extremely low prices at high mileage levels are visible.  
- **MMR Value**: Strong positive correlation with selling price, indicating MMR is a reliable market benchmark for pricing.

---

##### Correlation Heatmap (Numerical Features)

![Correlation Heatmap](../visualizations/Correlation_Heatmap.png)

**Insights:**

- **MMR and Selling Price**: Nearly perfect positive correlation (0.99).  
- **Odometer and Selling Price**: Moderately negative correlation (–0.58), suggesting depreciation with use.  
- **Year** also positively correlates with price—newer vehicles are more valuable.

---


#### Selling Price vs Categorical Features

##### Boxplots of Selling Price by Category

![Boxplots by Category](../visualizations/Boxplots_by_Category.png)

**Key Observations:**

- **Body Type**: Convertibles and coupes tend to have higher price ranges than minivans or hatchbacks.  
- **Brand (Make)**: Luxury brands like BMW and Infiniti show higher price distributions than others.  
- **Transmission**: Automatic vehicles dominate the dataset and exhibit a wide price spread.

---

##### Selling Price by Exterior Color

![Selling Price by Color Distribution](../visualizations/Selling_Price_by_Color_Distribution.png)

**Color-Based Behavior:**

- **Gray**, **white**, and **black** are the most common colors and span a wide price range.  
- Bright or niche colors (e.g., lime, turquoise) are associated with lower frequency and possibly more specialized pricing.

---

##### Selling Price by Interior Color

![Selling Price by Interior Density](../visualizations/Selling_Price_by_Interior_Density.png)

**Interior Preferences:**

- Black and beige interiors dominate volume.  
- **Off-white** shows a distinct price peak, indicating premium vehicle alignment.

---

##### Average Price by Exterior Color

![Average Price by Exterior Color](../visualizations/Average_Price_by_Exterior_Color.png)

**Insight:**

- **Charcoal**, **off-white**, and **black** tend to fetch higher average prices.  
- **Gold**, **white**, and **green** see lower average prices.

---

##### Average Price by Interior Color

![Average Price by Interior Color](../visualizations/Average_Price_by_Interior_Color.png)

**Observation:**

- **Off-white** interiors show a much higher average price—often a sign of luxury branding.  
- **Tan** and **blue** interiors correspond with lower average prices.

---
##### Top Brands with Off-White Interiors

![Top Brands with Off-White Interior](../visualizations/Top_Brands_with_Off-White_Interior.png)

**Insight:**

- **Mercedes-Benz** dominates the vehicles featuring off-white interiors, consistent with premium design choices.  
- Other luxury brands like **BMW** and **Lexus** appear but in much smaller counts.

---



#### Categorical vs Categorical Associations

##### Categorical Relationships (Cramér’s V)

![Cramér's V Heatmap](../visualizations/Cramer_V_Heatmap.png)

**Takeaways:**

- Strong association between **make** and **country**, and between **body** and **body_group**.  
- Most other categorical relationships are weak, suggesting minimal multicollinearity concerns.

---

### Time-based Trends

#### Monthly Sales Trend

![Monthly Vehicle Sales Trend](../visualizations/Monthly_Vehicle_Sales_Trend.png)

**Trend Observation:**

- Vehicle sales peaked around early 2015, indicating seasonal or market-driven patterns worth further business analysis.

---

### Summary of EDA

My exploratory analysis revealed several key insights:

- **Price Drivers**: Vehicle age (year), mileage (odometer), and market benchmark (MMR) are strongly associated with selling price, with MMR showing the highest correlation.
- **Category Patterns**: Body type, brand, and color exhibit notable price differences—luxury brands and niche colors often correlate with higher prices.
- **Market Skew**: Selling price, MMR, and odometer all show right-skewed distributions, suggesting the presence of high-end and high-mileage outliers.
- **Transmission Trends**: Automatic vehicles dominate the dataset and show a wide range in selling prices.
- **Temporal Variation**: Sales volume peaked in early 2015, potentially due to seasonal or economic trends.

These findings not only help shape our preprocessing and feature selection strategy, but also validate the dataset’s suitability for supervised learning and regression modeling.


---

## Data Cleaning and Preprocessing

---

### Duplicate & Missing Value Handling

- **Duplicate Check**: No duplicate rows found in the dataset.
- **Missing Value Check**:  
  - Columns with missing values were identified and visualized using bar charts, matrix charts, and heatmaps.
  - Low-missing columns (less than 1%) were handled via listwise deletion.
  - High-missing categorical columns were imputed using mode values.
  - Final check confirmed that missing values were fully addressed.

---

### Frequency & Categorical Value Checks

- **Frequency Check**: Categorical features were examined for unusual or rare values.
- **Discrete-Type Detection**: Identified variables with finite, interpretable value sets.
- **Categorical Frequency Summary**: Provided an overview of distribution across major categorical features.

---

### Categorical Feature Cleaning & Normalization

- **Column Standardization**:
  - `make` was mapped to its corresponding `country` (e.g., Ford → United States).
  - `body` types were grouped into body groups (e.g., coupe, sedan → Sedans and Coupes).
- **Text Normalization**:
  - Cleaned inconsistencies in `color` and `interior` columns (e.g., standardizing casing, removing whitespace).
  - Transmission values (e.g., 'A', 'Automatic') were encoded as binary.

---

### Outlier Detection

Box Plot Insights: Vehicle Features & Selling Price

1. **Model Year & Condition**
- **`year`**: Most vehicles are manufactured between **2010–2014**, with a few **older outliers** (pre-2000).
- **`condition`**: Fairly **evenly distributed**, centered around average values. No major outliers, suggesting condition scores are relatively stable across vehicles.

2. **Odometer**
- Shows a **heavily right-skewed** distribution.
- **Outliers above 300,000 miles** are prevalent, with some extreme values exceeding **900,000+ miles**.
- Consider **capping or trimming** the top 1% to reduce skewness and improve model robustness.

3. **MMR (Market-Based Reference Price)**
- Dominated by outliers above **$75,000**, with a long right tail.
- This feature likely represents **industry reference benchmarks** but contains values far beyond the majority.
- Consider applying **log transformation** or **capping** at a reasonable upper bound (e.g., 99th percentile).

4. **Selling Price**
- Highly **right-skewed** with numerous **extreme outliers** above **$100,000** and even **$200,000+**.
- Consider applying **log transformation** or **capping** at a reasonable upper bound (e.g., 99th percentile).

>Proposed Action Plan
>
>Apply **outlier treatment (e.g., quantile-based trimming)** or **transformation (e.g., log)** to `odometer`, `mmr`, and `sellingprice` to enhance model training and interpretability.


![Boxplots of Numerical Features](../visualizations/Boxplots_of_Numerical_Features.png)

**Insights:**

- **Odometer**, **MMR**, and **Selling Price** exhibit strong right-skewed distributions with substantial outliers.
- **Year** and **Condition** show relatively symmetric distributions with fewer extreme values.
- Visualizing these helps determine which variables need capping to improve model robustness.

---

### Outlier Summary (IQR-based Detection)

**Outlier Count and Percentage per Feature:**

- **mmr**: 2,523 outliers (0.56%)
- **sellingprice**: 2,361 outliers (0.52%)
- **odometer**: 424 outliers (0.09%)
- **year**: 0 outliers (0.00%)
- **condition**: 0 outliers (0.00%)

**Selected Features for Treatment**:  
`['mmr', 'sellingprice', 'odometer']`

**Key Insights:**

- Although **mmr** and **sellingprice** exhibit long tails, their outlier rates are under 1%, implying they are rare but potentially impactful.
- **odometer** has fewer outliers, yet these can still distort model training.
- **year** and **condition** are clean and stable with no detected outliers.

**Actionable Plan:**

- Either **retain** the outliers when using robust models (e.g., Gradient Boosting), or  
- Apply **capping** (e.g., 1st–99th percentile) to minimize the influence of extreme values and enhance model generalizability.

---

### Apply Capping to Selected Columns

#### Applied Method: 1st–99th Percentile Capping

To address extreme outliers without sacrificing data completeness, I applied **percentile-based capping**:

- **Method**: 1st–99th percentile capping  
- **Goal**: Limit the influence of extreme values while preserving all rows
- **Capping Applied To**:  `['mmr', 'sellingprice', 'odometer']`

#### Why 1st–99th Percentile?

- Retains **most data points**, ensuring downstream model performance is not compromised.
- Especially suitable for **right-skewed distributions** like price and mileage.
- Helps maintain **row count consistency**, which is crucial for dashboarding and time-based analyses.

#### Key Observations from Box Plots:

- **MMR and Selling Price**: Originally had long-tailed outliers, now capped to reflect more realistic market values.
- **Odometer**: Extremely high mileage vehicles (e.g., > 500,000 miles) were capped, improving scale uniformity.
- **Distribution Shape**: The core IQR structure remains intact, preserving meaningful variation for modeling.

**Outcome**:

- Outliers were effectively **capped** rather than removed.
- This approach **preserves row count and data structure**, ensuring minimal information loss while reducing the distortion from extreme values.
- Post-capping boxplots confirm that the majority of extreme outliers were compressed into the whisker range.

![Before vs After Capping](../visualizations/Before_vs_After_Capping_Boxplots.png)

**Takeaways:**

- After applying percentile-based capping (1st–99th percentile), extreme outliers are removed without distorting the core distribution.
- This transformation reduces the influence of rare but extreme values in **mmr**, **odometer**, and **sellingprice**, which is crucial for stable model training.

---

### One-Hot Encoding

To prepare the dataset for machine learning algorithms, I applied **One-Hot Encoding** to convert categorical variables into binary features.

#### Feature Engineering Steps:

- Converted `saledate` to datetime format and extracted **weekday** (`1–7`) as a new temporal feature.
- Replaced original columns for better semantic clarity:
  - `make` was replaced with `country` to focus on **vehicle origin**.
  - `body` was replaced with `body_group` to generalize **vehicle types**.

#### Encoded Categorical Columns:

- `body` → One-hot encoded as `Cat_*`
- `make` (country) → One-hot encoded as `Brand_*`
- `color` (exterior) → One-hot encoded as `C_*`
- `interior` (interior material) → One-hot encoded as `I_*`

#### Output:

- **Dummy columns** were cast to `int64` for full compatibility with ML models.
- **Final dataset shape**: e.g., `(451,244, 58)` — includes all one-hot encoded features.

One-hot encoding expands categorical features into numeric binary vectors, enabling regression and tree-based models to capture category effects.

#### Final Encodings Summary

- One-hot encoding was applied to key categorical columns: `body`, `make`, `color`, and `interior`.
- Ensured all dummy variables were converted to `int64` for model compatibility.
- A new feature `weekday` was created from `saledate` to capture potential day-of-week effects.

Final cleaned dataset saved as: `one_hot_cleaned_car_prices_for_modeling.csv`

---

### Feature Selection and Preparation

---

#### Retained Features (10)

To balance model accuracy and interpretability, we selected the top 10 most relevant features based on exploratory analysis, box plots, and correlation heatmaps:

| Feature             | Description                                               |
|---------------------|-----------------------------------------------------------|
| `year`              | Newer vehicles usually command higher resale value        |
| `condition`         | Better condition positively correlates with price         |
| `odometer`          | Higher mileage generally reduces price                    |
| `mmr`               | Manheim Market Report value – a strong price benchmark    |
| `Cat_Pickup Trucks` | Key body type category                                    |
| `Cat_Sedans and Coupes` | Another major body type segment                      |
| `Brand_Germany`     | Captures premium brand influence (e.g., BMW, Mercedes)    |
| `C_black`           | Most common exterior color                                |
| `I_black`           | Dominant interior color                                   |
| `I_gray`            | Popular interior tone                                     |

These variables, along with the target variable `sellingprice`, were retained to form the final modeling dataset. All other variables were dropped to reduce noise and multicollinearity.

---

#### Final Feature Correlation Heatmap

![Correlation Matrix of Selected Features](../visualizations/Selected_Feature_Correlation_Heatmap.png)

**Insights:**

- **MMR (0.98)** shows an almost perfect linear relationship with `sellingprice`.
- **Year (0.61)** and **condition (0.55)** also contribute significantly.
- **Odometer** presents a strong negative relationship (–0.62), aligning with depreciation trends.
- Other categorical indicators (e.g., color, brand, body) have moderate but meaningful associations.

These findings validate the selected features as meaningful predictors of vehicle price.


---


### Formulating Business Analytics Questions

---

#### 1. Which factors most significantly influence the selling price of used vehicles?

**Why it matters:**  
Understanding what drives pricing helps dealers, resellers, and analysts make smarter pricing and sourcing decisions. It also increases transparency for customers.

**How it connects to the data:**  
By analyzing model-based feature importance (e.g., from Gradient Boosting or XGBoost), we can identify which variables—such as `mmr`, `condition`, `odometer`, `year`, or `brand`—most affect price.

**Expected outcome:**  
A ranked list of influential features based on their predictive importance in determining vehicle selling price.

---

#### 2. How accurately can the selling price of a vehicle be predicted based on its features?

**Why it matters:**  
Accurate predictions support automated pricing tools, better inventory valuation, and competitive strategy.

**How it connects to the data:**  
Using detailed features like `mmr`, `condition`, and `brand`, regression models can be trained and evaluated. Model performance will be assessed using metrics such as R², RMSE, and MAE.

**Expected outcome:**  
A summary of prediction accuracy and comparison across top-performing models (e.g., Gradient Boosting, LightGBM, XGBoost), validating overall reliability.

---

#### 3. How well does the predictive model generalize across different vehicle categories or brands?

**Why it matters:**  
Consistent performance across categories is critical for business deployment. Uneven accuracy may lead to unfair pricing or business risk.

**How it connects to the data:**  
By segmenting results by vehicle body type (e.g., SUV vs Other) or brand origin (e.g., Japanese vs Others), we can analyze accuracy trends and residuals across subgroups.

**Expected outcome:**  
Insights into where the model performs best or needs refinement, ensuring balanced performance across vehicle segments.

---

## Predictive Modeling 

---

### Defining Functions: Model Selection, Validation & Comparison

I defined reusable functions to automate model training, cross-validation, evaluation, and comparison across multiple regressors. These functions help streamline the modeling workflow and ensure consistency.

- `evaluate_models_pipeline()`  
- `compare_and_select_best_models()`  
- `cross_validate_model()`
- `compare_and_select_best_models()`
- `save_model_with_params()`
- `evaluate_models_pipeline()`
- `SmartModelSelector` 

These tools support both baseline and tuned models, and provide metrics such as **R²**, **MAE**, **MSE**, and **RMSE** for comparison.

---

### Feature Selection and Preparation

#### Data Checking and Reloading

Before feature selection, I checked the cleaned dataset to ensure all preprocessing steps were applied correctly and no transformations were missing.

#### Lasso Regression for Feature Importance

To select the most predictive features, I used **Lasso Regression**:

- Automatically penalizes less important variables by shrinking their coefficients to zero
- Helps improve generalization by reducing overfitting

Key steps:

- Standardized input features
- Used cross-validation to determine optimal regularization strength (`alpha`)
- Visualized Lasso coefficients to guide final selection

#### Final Feature Selection

Based on Lasso results and domain relevance, I selected the following 8 features for final modeling:

🔹 Selected Features:
1. year
2. Cat_SUVs and Crossovers
3. odometer
4. weekday
5. Brand_Japan
6. C_white
7. condition
8. mmr

These features, combined with the target `sellingprice`, form the final modeling dataset.
The final modeling dataset, containing the selected 10 key features and the target variable `sellingprice`, was saved as: final_selected_car_prices_for_modeling.csv

### Data Preparation for Modeling

#### Data Checking Before Selecting

To prevent data mismatch or loss due to session interruptions (e.g., in Colab environments), I reloaded and revalidated the final dataset before proceeding with feature selection and scaling.

This step ensured:

- No shape inconsistencies
- No unexpected missing values
- All selected features remained available and correctly typed

By doing so, I guaranteed a clean and stable base for downstream modeling tasks.

#### Manual Feature Set Selection (Optional Step)

Although the final modeling used features selected via Lasso and correlation analysis, I also implemented an optional manual selection step.

**Purpose:**

- To allow flexibility for modeling with alternative datasets or domain-specific feature sets.
- To provide a checkpoint where business logic or new hypotheses can guide feature refinement.

This step is especially helpful for experimenting with different feature combinations in future iterations or business-specific models.

#### Data Scaling & Train-Test Split

- Applied MinMaxScaler / StandardScaler to normalize numeric features
- Split dataset into **training (80%)** and **test (20%)** subsets with `random_state=42` for reproducibility

This dataset is fully preprocessed and feature-selected, ready for training and evaluation across various regression models.

---

### Unified Modeling and Model Comparison

---

#### Overview

To establish a benchmark and select the most robust regression model, I trained and evaluated six models:

- Linear Regression  
- Random Forest  
- Gradient Boosting  
- HistGradientBoosting  
- LightGBM  
- XGBoost  

Each model was assessed based on training and test performance metrics, including R², MSE, RMSE, and MAE. The goal was to identify high-performing models that generalize well and avoid overfitting.


| Model                | Train R² | Test R² | Test RMSE | Overfit Gap |
|---------------------|----------|---------|------------|--------------|
| **LightGBM**         | 0.9764   | 0.9763  | 0.1539     | **0.0001**   |
| HistGradientBoosting| 0.9763   | 0.9762  | 0.1540     | 0.0001       |
| XGBoost             | 0.9774   | 0.9760  | 0.1548     | 0.0014       |
| Gradient Boosting   | 0.9756   | 0.9757  | 0.1556     | -0.0001      |
| Linear Regression   | 0.9746   | 0.9747  | 0.1595     | -0.0001      |
| Random Forest       | 0.9963   | 0.9738  | 0.1618     | **0.0225**   |

---

#### Test Performance Comparison (Bar Charts)

To visually compare model generalization on unseen data, I plotted four bar charts based on test set results:

![Test Performance Comparison](../visualizations/Model_Comparison_Test_Metrics.png)

**Observations:**

- **Test R²**: LightGBM, HistGradientBoosting, and XGBoost show the highest explanatory power.
- **Test MSE & RMSE**: LightGBM leads with the lowest error, closely followed by HistGradientBoosting.
- **Test MAE**: All boosting models outperform linear and random forest in absolute prediction accuracy.

These charts reinforce the earlier ranking and help validate model consistency across different error metrics.

---

#### Performance Summary, Overfitting and Generalization Analysis

![Overfitting and R2 Comparison](../visualizations/R2_Overfitting_Comparison.png)

**Insights:**

- **LightGBM** and **HistGradientBoosting** show the best balance between accuracy and generalization, with minimal overfit.
- **Random Forest** achieves a high training R² but overfits significantly, as seen from the large test gap.
- **Gradient Boosting** and **Linear Regression** perform consistently with small gaps and stable R² values.
- **XGBoost** performs well, but with a slightly higher overfit gap than LightGBM.

**LightGBM** was selected as the most balanced and reliable model, offering excellent predictive power with minimal overfitting risk.

All models were evaluated using `SmartModelSelector` and `evaluate_models_pipeline()` to ensure consistent comparison and metric tracking across training and testing phases.


### Per-Model Fine-Tuning

#### Model Tuning and Selection Strategy

I trained and evaluated six regression models:

- Linear Regression  
- Random Forest  
- Gradient Boosting  
- LightGBM  
- XGBoost  
- HistGradientBoosting  

All models underwent baseline training and multi-stage hyperparameter tuning (Halving Search → Randomized Search → Grid Search).  
However, to maintain clarity and focus in this report, I present **detailed analysis only for the top-performing models**, while briefly summarizing the rest in the comparison section.


#### Selected Models for In-Depth Analysis

Among all six trained models, three stood out in terms of both predictive accuracy and generalization performance:

- **Gradient Boosting Regressor (GBR)** achieved the **highest R² (0.9764)** with very low RMSE and MAE.
- **LightGBM (GridSearch)** matched GBR in performance while maintaining the smallest overfit gap.
- **HistGradientBoosting (HistGB)** demonstrated **excellent generalization** with the **lowest overfit gap (0.0002)**.

These models are selected for detailed interpretation in the following sections.

While Gradient Boosting delivered the highest R² score (0.9764), **LightGBM achieved nearly identical performance** (R² = 0.9763, RMSE = 0.1536, MAE = 0.1008) with one key advantage: **speed**.

- **Training Efficiency**: LightGBM uses a histogram-based algorithm and leaf-wise tree growth, making it **much faster** than traditional Gradient Boosting.
- **Scalability**: It handles large datasets efficiently, making it suitable for real-world production environments.
- **Overfitting Control**: The model maintained a low overfit gap (0.0008), demonstrating strong generalization.

Given its balance of **accuracy, speed, and robustness**, LightGBM is a **top choice for deployment and further optimization**.

### LightGBM Modeling Workflow

### 1. Baseline Model — LightGBM Regressor

A baseline LightGBM model was trained with default parameters to establish a performance reference. Despite no tuning, the model achieved strong R² and MAE scores, confirming LightGBM’s reliability on this dataset.

- **Train R²**: 0.9764  **Test R²**: 0.9763  
- **Test MAE**: 0.1011  **Test RMSE**: 0.1539  

The following plot shows the model's predictions vs. actual selling prices:

![Actual vs Predicted — LightGBM Baseline](../visualizations/LightGBM_Baseline_Actual_vs_Predicted.png)

**Insight:**  
Predictions align closely with actual values, clustering around the diagonal line. Minor deviations appear at extreme values but are acceptable at baseline.


---

### 2. Feature Importance — LightGBM Regressor

Before hyperparameter tuning, a feature importance analysis was conducted to understand which features most strongly influence vehicle price prediction.

![All Feature Importances (LightGBM)](../visualizations/LightGBM_Feature_Importance.png)

**Insights:**

- **mmr (Manheim Market Report)** is by far the most influential feature, confirming its role as a key pricing benchmark.
- **condition**, **odometer**, and **year** also contribute significantly, aligning with expectations around vehicle quality, usage, and age.
- **weekday**, **color (C_white)**, and **origin (Brand_Japan)** show minor impact but may still capture niche effects or trends.

This analysis guided the refinement of the final feature set for tuning and final model training.

---

### 3. Hyperparameter Tuning — LightGBM Regressor

**LightGBM Hyperparameter Tuning Strategy**

To optimize the LightGBM model, I designed a **three-stage tuning strategy** to balance speed, accuracy, and model generalization. Each stage progressively refines the hyperparameters.

#### 3.1 Hyperparameter Tuning — HalvingRandomSearchCV

To improve the baseline LightGBM model, I first performed hyperparameter tuning using **HalvingRandomSearchCV** for a coarse-grained hyperparameter search, which offers an efficient way to explore hyperparameter space with fewer resources.

This approach progressively narrows the search space by allocating more resources to promising configurations.

```python
lgb_halving_base_params = lgb_baseline_model.get_params()

param_dist_lgb_halving = {
    'max_depth': list(set([lgb_halving_base_params['max_depth'], 10, 20, -1])),
    'num_leaves': sorted(list(set([lgb_halving_base_params['num_leaves'], 15, 63]))),
    'min_child_samples': sorted(list(set([lgb_halving_base_params['min_child_samples'], 10, 30]))),
    'learning_rate': sorted(list(set([lgb_halving_base_params['learning_rate'], 0.05, 0.01])))
}

**Evaluation Result:**

| Metric        | Train        | Test         |
|---------------|--------------|--------------|
| R² Score      | 0.9767       | 0.9762       |
| MSE           | 0.0234       | 0.0237       |
| RMSE          | 0.1528       | 0.1539       |
| MAE           | 0.1005       | 0.1011       |

**Observation:**

- Model performance is nearly identical to the baseline, but slightly improved in RMSE and MAE.
- Overfitting remains minimal, confirming tuning stability.

This version was used as the starting point for further fine-tuning using RandomizedSearchCV.

#### 3.2 Hyperparameter Tuning — RandomizedSearchCV

After identifying promising regions via Halving Search, I conducted a more refined search using **RandomizedSearchCV**. This strategy randomly samples parameter combinations from predefined distributions, offering a balance between search breadth and resource efficiency.

```python
lgb_random_base_params = lgb_halving_best_model.get_params()

param_dist_random = {
    'max_depth': list(set([lgb_random_base_params['max_depth'], 10, 20, -1])),
    'num_leaves': sorted(list(set([lgb_random_base_params['num_leaves'], 15, 63]))),
    'min_child_samples': sorted(list(set([lgb_random_base_params['min_child_samples'], 10, 30]))),
    'learning_rate': sorted(list(set([lgb_random_base_params['learning_rate'], 0.01, 0.05, 0.1]))),
    'n_estimators': randint(50, 300)
}

**Evaluation Result:**

| Metric        | Train        | Test         |
|---------------|--------------|--------------|
| R² Score      | 0.9771       | 0.9763       |
| MSE           | 0.0229       | 0.0236       |
| RMSE          | 0.1514       | 0.1537       |
| MAE           | 0.0997       | 0.1008       |

**Observation:**

- Slight improvement across all metrics, especially in MAE and MSE.
- Excellent generalization with minimal overfitting.
- Performance is now marginally better than both baseline and Halving models, supporting the effectiveness of this tuning stage.

Next, I proceeded with a final fine-tuning using GridSearchCV to further optimize the parameters.

#### 3.3 Hyperparameter Tuning — GridSearchCV

To finalize the model, I performed an exhaustive search using **GridSearchCV**, building upon the best parameters obtained from the Randomized Search stage. This method tests all combinations within a tight parameter neighborhood to find the optimal configuration.

```python
lgb_grid_base_params = lgb_random_best_model.get_params()

lgb_grid_param = {
    'learning_rate': [lgb_grid_base_params['learning_rate'] * f for f in [0.8, 1.0, 1.2]],
    'num_leaves': [
        max(2, lgb_grid_base_params['num_leaves'] - 5),
        lgb_grid_base_params['num_leaves'],
        lgb_grid_base_params['num_leaves'] + 5
    ],
    'min_child_samples': [
        max(1, lgb_grid_base_params['min_child_samples'] - 5),
        lgb_grid_base_params['min_child_samples'],
        lgb_grid_base_params['min_child_samples'] + 5
    ]
}

**Tuning Highlights**

- Focused grid constructed around the best randomized parameters.
- Slightly adjusted **`learning_rate`** (±20%) to explore trade-offs between convergence speed and generalization.
- Fine-tuned **`num_leaves`** and **`min_child_samples`** to balance model complexity and robustness.

---

** Performance Summary (GridSearchCV Result):**

| Metric       | Value     |
|--------------|-----------|
| **Train R²**     | 0.977140  |
| **Test R²**      | 0.976347  |
| **Train RMSE**   | 0.151239  |
| **Test RMSE**    | 0.153610  |
| **Test MAE**     | 0.100805  |
| **Overfit Gap**  | ~0.0008   |

---

This final tuning stage yielded the most balanced performance in terms of **accuracy** and **generalization**, confirming **LightGBM** as the best model for the task.

### 4. Save and Restore

To ensure reproducibility and avoid re-training:

- The final model and tuning results were **saved using `joblib`**.
- Reload tested and verified for consistency across sessions.

---

### 5. LightGBM Model Performance Comparison

Compared tuned LightGBM against:

- Baseline LightGBM
- Gradient Boosting
- XGBoost
- Linear Regression, Random Forest, etc.

**Result**: Tuned LightGBM performed nearly best overall, **with the smallest overfitting gap** and high test R² (0.9763).

---

### 6. Final Model Selection

Selected **LightGBM (GridSearch-tuned)** as the final model for deployment and interpretation due to:

- High accuracy
- Low overfitting
- Efficient training time

---

### 7. Feature Importance + Residual Analysis (LightGBM)

Visualized:

- **Feature Importance**: Confirmed MMR and odometer are key.
- **Residual Plots**: No major pattern or bias detected — errors centered and stable.
- **Prediction vs Actual Scatterplot**: Tight alignment confirmed model validity.

---

### 8. LightGBM Model Interpretation

- **Business Interpretation**: The model appropriately emphasizes market (MMR), condition, usage (odometer), and brand cues (e.g., German brands).
- **Practical Use**: Can be deployed in automated pricing engines or decision dashboards with confidence.

---














### 1. Variable Distributions

- **Vehicle Year**: Most vehicles are manufactured between 2007 and 2015, with a peak around 2012.
- **Condition**: While condition is numeric, it's not a typical 1–5 scale; values range widely, suggesting different rating schemes from various sellers.
- **Odometer**: Positively skewed; many vehicles cluster under 100,000 miles, but some outliers reach nearly 1,000,000 miles.
- **Selling Price**: Right-skewed distribution; majority of cars sold between \$5,000 and \$20,000, but luxury vehicles can exceed \$100,000.

> 📊 *Included plots: histograms for year, condition, odometer, and sellingprice.*


---

### 4. Missing Data Analysis

- **Transmission**: ~11% missing
- **Condition**: ~2% missing
- **Body/Trim/Model**: Some inconsistencies and null values exist but affect <5% of entries

> ✅ Missing values handled later during preprocessing via imputation or removal.

---

### 5. Outlier Detection

- **MMR, Selling Price, and Odometer** contain extreme values (e.g., prices over \$200,000; odometers close to 1,000,000).
- Outliers detected using IQR and capped at 1st–99th percentile range to reduce skew without removing rows.

> 📦 *Visuals: boxplots and before/after capping distributions.*

---

### Summary of Key Findings

- The dataset is rich, but contains noise and outliers, which need careful preprocessing.
- MMR, odometer, and condition are strong predictors for price.
- Sedans and SUVs dominate the market, with SUVs commanding higher average prices.
- Japanese brands tend to show more consistent pricing behavior.



