# Customer Churn Prediction — End-to-End ML Pipeline

## Business Problem

ConnectTel is facing a significant challenge with customer churn, which impacts its revenue and growth. The objective of this project is to develop an end-to-end machine learning (ML) system for predicting customer churn, combining exploratory analysis, feature engineering, class-imbalance-aware modeling, and threshold optimisation into a reproducible training pipeline. This robust customer churn prediction system uses advanced analytics and machine learning (ML) techniques to help identify customers at high risk of churn, understand key churn drivers, and provide actionable insights for retention strategy. By accurately forecasting which customers are likely to churn, ConnectTel can implement targeted retention strategies to reduce customer attrition, enhance loyalty, and maintain a competitive edge in the telecommunications industry. This project goes beyond “train a classifier”. It investigates why customers churn, builds features informed by domain insights, and packages the workflow into deployable artifacts.

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
Churn is minority and requires PR-based evaluation and threshold tuning.

**B. Strong Drivers of Churn**

**1. Contract Type**
- Month-to-month contracts has the highest churn rate
- Two-year contracts has the lowest churn rate
- Longer contracts strongly reduce churn risk

***Correlation with churn: -0.32***

Tenure

Short-tenure customers churn significantly more

Longer tenure strongly reduces churn likelihood

Correlation with churn: -0.35

Internet Service

Fiber optic customers show higher churn

DSL churn moderate

No-internet lowest churn

Payment Method

Electronic check → highest churn

Bank transfer & credit card → lower churn

Billing & Charges

Higher MonthlyCharges → higher churn (correlation: 0.19)

Paperless billing → associated with higher churn

MonthlyCharges & TotalCharges strongly correlated (0.65)

Demographics

Senior citizens slightly more likely to churn

Customers without partners or dependents churn more

C. Engineered Features

Based on EDA:

TotalServices → count of subscribed services

SeniorCitizenCat → categorical version for modeling consistency

EDA insights inform modeling but no leakage occurs — all transformations are inside the sklearn pipeline.





### Feature Engineering
- **Encoding Categorical Variables:** Convert categorical variables into numerical format using Label Encoding and One-Hot Encoding.
- **Creating New Features:** Generate new features based on insights from EDA to enhance model performance.

### Model Selection, Training, and Validation
- **Train-Test Split:** Split the data into training and testing sets.
- **Model Training:** Train three supervised learning models, including:
  - Logistic Regression
  - Gradient Boosting
  - Support Vector Machine (SVM)
- **Hyperparameter Tuning:** Use RandomizedSearchCV to optimise model hyperparameters.
- **Model Evaluation:** Evaluate model performance using the following metrics precision, recall, F1-score and confusion matrix.

### Model Evaluation
- **Compare Models:** Analyse the results of the models and select the best-performing one based on evaluation metrics.
- **Business Considerations:** Consider the impact of false positives and false negatives on the business to decide the best-suited model.

**Model Interpretation and Insights**
- Interpret the model results to derive actionable insights.
- Identify the most significant features that influence customer churn.

## Challenges Faced
- Handling class imbalance in the dataset.
- Feature engineering to improve model performance.
- Selecting the most appropriate evaluation metrics for the business problem.

## Conclusion

ConnectTel can proactively address customer attrition and implement effective retention strategies by developing an accurate customer churn prediction model. This project demonstrates the application of data science and machine learning techniques to solve a critical business problem in the telecommunications industry.

### Dependencies

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

**Collaborations:**

You can reach out to me using my contact below for gigs and collaborations.

**Contact:**

ike@ikemefulaoriaku.space | (https://www.linkedin.com/in/gentleiyke/)
