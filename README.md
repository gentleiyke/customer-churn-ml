# Customer Churn Prediction — End-to-End ML Pipeline

## Business Problem

ConnectTel was facing a significant challenge with customer churn, which impacts its revenue and growth. The objective of this project is to develop an end-to-end machine learning (ML) system for predicting customer churn, combining exploratory analysis, feature engineering, class-imbalance-aware modeling, and threshold optimisation into a reproducible training pipeline. 

This robust customer churn prediction system uses advanced analytics and machine learning (ML) techniques to help identify customers at high risk of churn, understand key churn drivers, and provide actionable insights for retention strategy. 

By accurately forecasting which customers are likely to churn, ConnectTel can implement targeted retention strategies to reduce customer attrition, enhance loyalty, and maintain a competitive edge in the telecommunications industry. This project goes beyond “train a classifier”. It investigates why customers churn, builds features informed by domain insights, and packages the workflow into deployable artifacts.

**Target variable:** Churn
**Positive class = customers who left = 1**

## Dataset Overview
The dataset contains 7,043 customers, 21 original features, and a mix of demographic, service, billing, and account variables. Key groups are:
- **Demographics:** (gender, senior status, dependents)
- **Service subscriptions:** (internet type, add-ons)
- **Contract & billing:** (contract type, payment method, paperless billing)
- **Charges:** (MonthlyCharges, TotalCharges)
- **Tenure:** (months with company)
- **Target variable:** (Churn) - ***Positive class = customers who left = 1***

### Exploratory Data Analysis (EDA)
The notebook (Customer_Churn_Analysis.ipynb) performs deep EDA before modeling.
**Key Insights**

**A. Class Imbalance**

- Majority class = non-churn

*Churn is minority and requires PR-based evaluation and threshold tuning.*

**B. Strong Drivers of Churn**

**1. Contract Type**
- Month-to-month contract customers has the highest churn rate
- Two-year contract customers has the lowest churn rate

*Longer contracts strongly reduce churn risk (correlation with churn: -0.32)*

**2. Tenure**
- Short-tenure customers churn significantly more
- Longer tenure strongly reduces churn likelihood

*Correlation with churn: -0.35*

**3. Internet Service**
- Fiber optic customers show higher churn
- Customers with DSL churn shown moderate churn
- Customers with no internet show the lowest churn

**4. Payment Method**
- Electronic check customers show the highest churn
- Bank transfer and credit card customers show lower churn

**5. Billing & Charges**
- Customers with higher monthly charges show higher churn (*correlation: 0.19*)
- Paperless billing customers are associated with higher churn

*Monthly Charges and Total Charges strongly correlated (0.65)*

**6. Demographics**
-  Customers who are senior citizens are slightly more likely to churn
-  Customers without partners or dependents churn more

**C. Engineered Features (Based on EDA)**
- TotalServices: count of subscribed services
- SeniorCitizenCat: categorical version for modeling consistency

*EDA insights inform modelling, but no leakage occurs — all transformations are inside the sklearn pipeline.*

### Modeling Strategy
**A. Preprocessing:** implemented a leakage-safe pipeline using sklearn Pipeline for 
- Numeric Variables: to scale varaiables and handle missing values using median imputation
- Categorical Variables: to implement one-hot encoding and handle missing values using most frequent imputation
- Combining pipelines using ColumnTransformer

**B. Candidate Models:** 
- Logistic Regression
- SVC
- Histogram Gradient Boosting

**C. Model Selection Strategy** 
- 5-fold cross-validation on training data
- Compare models using ROC-AUC and PR-AUC
- Select best model using validation PR-AUC
- Optimise classification threshold using validation F1
- Refit on train and validation
- Final evaluation on unknown test set

### Final Results (Test Set)

Best model: Logistic Regression (balanced)
Selected threshold: 0.61

**Core Performance Metrics**
| Metric    | Value |
| --------- | ----- |
| ROC-AUC   | 0.842 |
| PR-AUC    | 0.633 |
| Precision | 0.546 |
| Recall    | 0.701 |
| F1 Score  | 0.614 |

**Confusion Matrix (Test Set)**
|                | Predicted No Churn | Predicted Churn |
| -------------- | ------------------ | --------------- |
| **Actual No**  | 817                | 218             |
| **Actual Yes** | 112                | 262             |

**Interpretation**
- Threshold tuning improved recall while maintaining usable precision
- Recall prioritised to catch churners
- Balanced precision–recall tradeoff

### Repos Structure
```
artifacts/
  ├── churn_model.joblib
  ├── metrics.json
  ├── schema.json
  ├── cv_results.csv
  ├── val_results.csv
  ├── threshold_sweep.csv
  └── pr_curve.csv
data/ 
  └── Customer-Churn.csv
notebooks/ 
  └── Customer_Churn_Analysis.ipynb
src/
  ├── config.py
  ├── data.py
  ├── pipeline.py
  ├── evaluate.py
  └── train.py
README.md
requirements.txt
```

**How to Run**
```
python train.py --data-path Customer-Churn.csv --artifact-dir artifacts
```

Dependencies

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Missingno
- Joblib

**Collaborations:**

You can reach out to me using my contact below for gigs and collaborations.

**PM:**

ikemefulaoriaku@gmail.com | (https://www.linkedin.com/in/gentleiyke/)
