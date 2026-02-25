# Customer Churn Prediction — End-to-End ML Pipeline

## Business Problem

ConnectTel is facing a significant challenge with customer churn, which impacts its revenue and growth. The objective of this project is to develop an end-to-end machine learning (ML) system for predicting customer churn, combining exploratory analysis, feature engineering, class-imbalance-aware modeling, and threshold optimisation into a reproducible training pipeline. This robust customer churn prediction system uses advanced analytics and machine learning (ML) techniques to help identify customers at high risk of churn, understand key churn drivers, and provide actionable insights for retention strategy. By accurately forecasting which customers are likely to churn, ConnectTel can implement targeted retention strategies to reduce customer attrition, enhance loyalty, and maintain a competitive edge in the telecommunications industry.

Customer Churn Prediction — End-to-End ML Pipeline

An end-to-end ML system for predicting customer churn, combining exploratory analysis, feature engineering, class-imbalance-aware modeling, and threshold optimisation into a reproducible training pipeline.

This project goes beyond “train a classifier.” It investigates why customers churn, builds features informed by domain insights, and packages the workflow into deployable artifacts.

ConnectTel faces revenue loss due to customer churn.
The objective is to:

identify customers at high risk of churn, understand key churn drivers, and provide actionable insights for retention strategy

Target variable: Churn
Positive class = customers who left (1)


## Dataset

The dataset consists of customer information, including demographic details, account information, service usage patterns, and churn status. The features include:

- **Demographic Information:** Gender, SeniorCitizen, Partner, Dependents
- **Account Information:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Service Information:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Churn Information:** Churn

### Problem Definition
- **Objective:** Develop a machine learning model to predict customer churn.
- **Target Variable:** Churn (Yes/No)

### Data Exploration and Preprocessing
- **Load Libraries:** Import necessary libraries for data analysis, visualization, and machine learning.
- **Data Loading:** Load the dataset and examine its structure.
- **Data Cleaning:** Handle missing values and drop unnecessary columns.
- **Exploratory Data Analysis (EDA):**
  - Visualize relationships between the target variable and key features.
  - Explore correlations and conduct univariate, bivariate, and multivariate analysis.

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
