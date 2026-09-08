# CreditWise — Loan Approval Prediction

A Machine Learning classification project that predicts whether a loan application is **Approved** or **Rejected** using applicant, financial, employment, and credit-related information.

---

## 📌 Overview

CreditWise is an end-to-end supervised Machine Learning classification project developed to predict loan approval outcomes from historical loan application data.

The project covers the complete workflow from data understanding and exploratory analysis to preprocessing, feature engineering, model comparison, hyperparameter tuning, and final model selection.

Three classification algorithms were evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes**

The models were evaluated using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

---

## 🎯 Problem Statement

Loan applications are evaluated using factors such as income, employment status, credit score, existing loans, debt-to-income ratio, savings, collateral, and loan requirements.

The objective of CreditWise is to build a Machine Learning system that can assist in predicting whether a loan application should be **Approved** or **Rejected** based on historical application data.

Because incorrect approval decisions can have financial consequences, the project considers multiple evaluation metrics rather than relying only on accuracy.

---

## 📊 Dataset

The original dataset contains:

- **1,000 loan applications**
- **20 columns**

The dataset contains both numerical and categorical information about applicants and their loan applications.

Rows with missing values in the target variable `Loan_Approved` were removed because a supervised learning model requires a known target label.

After this target cleaning step, the dataset contained **950 labeled records**.

### Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Applicant income |
| `Coapplicant_Income` | Coapplicant income |
| `Employment_Status` | Employment status |
| `Age` | Applicant age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Applicant savings |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property area |
| `Education_Level` | Applicant education level |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |

### Target Variable

`Loan_Approved`

```text
1 → Approved
0 → Rejected
```

---

## 🔎 Exploratory Data Analysis

### Loan Approval Distribution

![Loan Approval Distribution](images/target_distribution.png)

After removing records with missing target values, approximately **68.6%** of applications were labeled as Rejected and **31.4%** as Approved.

This distribution was considered when interpreting model performance.

### Correlation Analysis

![Correlation Heatmap](images/correlation_heatmap.png)

Correlation analysis was used to examine the relationship between numerical features and the loan approval outcome.

Among the numerical features, **Credit Score** showed the strongest positive relationship with loan approval, while **DTI Ratio** showed the strongest negative relationship.

### Feature Distributions

![Feature Distributions](images/feature_distributions.png)

The distributions of important numerical variables were examined to understand their spread, scale, and general behavior before model development.

---

## 🔄 Machine Learning Workflow

```text
Data Loading
     ↓
Raw Data Inspection
     ↓
Exploratory Data Analysis
     ↓
Remove Rows with Missing Target
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Feature Type Detection
     ↓
Data Preprocessing
     ├── Numerical Imputation
     ├── Numerical Scaling
     ├── Categorical Imputation
     └── One-Hot Encoding
     ↓
Baseline Model Training
     ├── Logistic Regression
     ├── KNN
     └── Gaussian Naive Bayes
     ↓
Prediction & Evaluation
     ↓
Feature Engineering
     ↓
Retraining & Re-evaluation
     ↓
Hyperparameter Tuning
     ↓
Final Model Comparison
     ↓
Best Model Selection
```

---

## ⚙️ Data Preprocessing

Different preprocessing operations were applied based on feature type.

### Numerical Features

Missing numerical values were handled using **mean imputation**, followed by standardization using `StandardScaler`.

```text
Numerical Features
       ↓
SimpleImputer
       ↓
StandardScaler
```

### Categorical Features

Missing categorical values were handled using **most-frequent imputation**, followed by One-Hot Encoding.

```text
Categorical Features
       ↓
SimpleImputer
       ↓
OneHotEncoder
```

A `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical features.

The preprocessing stage was integrated into the model pipelines so that preprocessing parameters are learned from the training data and then applied to the test data.

---

## 🤖 Models Evaluated

### Logistic Regression

Logistic Regression was used as a linear classification model for predicting loan approval outcomes.

### K-Nearest Neighbors (KNN)

KNN classifies observations based on the classes of nearby training observations. Feature scaling is important because KNN relies on distances between observations.

### Gaussian Naive Bayes

Gaussian Naive Bayes was used as a probabilistic classification approach based on the estimated feature distributions within each class.

---

## 📏 Evaluation Metrics

The following metrics were used to evaluate the models:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### Why multiple metrics?

Accuracy provides the overall proportion of correct predictions, but it does not explain the types of classification errors.

For this loan approval problem, **Precision** is particularly relevant when the goal is to reduce false-positive approval decisions.

With:

```text
1 → Approved
0 → Rejected
```

a false positive means:

```text
Actual:    Rejected
Predicted: Approved
```

Therefore, precision for the Approved class helps measure how often approval predictions were actually correct.

At the same time, Recall is important because it measures how many actual approved applications were correctly identified.

For overall model selection, the **F1 Score** was used as the primary comparison metric because it balances Precision and Recall.

---

## 🧪 Feature Engineering

Two nonlinear features were introduced:

- `DTI_Ratio²`
- `Credit_Score²`

The purpose was to test whether nonlinear transformations of DTI Ratio and Credit Score could provide additional predictive information.

The models were retrained and evaluated after introducing these engineered features.

### Performance Before vs. After Feature Engineering

| Model | Baseline F1 | Feature-Engineered F1 | Improvement |
|---|---:|---:|---:|
| Logistic Regression | 81.74% | 82.76% | +1.02% |
| KNN | 69.64% | 79.34% | +9.70% |
| Gaussian Naive Bayes | 78.18% | 82.93% | +4.75% |

Feature engineering improved the F1 Score of all three models, with the largest improvement observed for KNN.

This suggests that the engineered nonlinear features provided additional useful information for the classification models.

---

## 🔧 Hyperparameter Tuning

After feature engineering, hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

### Best Hyperparameters

| Model | Hyperparameter | Best Value |
|---|---|---:|
| Logistic Regression | `C` | `1` |
| KNN | `n_neighbors` | `15` |
| Gaussian Naive Bayes | `var_smoothing` | `1e-11` |

The purpose of tuning was to identify hyperparameter values that produced the strongest cross-validation performance.

---

## 📊 Final Model Comparison

![Final Model Performance Comparison](images/model_comparison.png)

The final results after feature engineering and hyperparameter tuning were:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **90.00%** | 81.54% | **88.33%** | **84.80%** |
| KNN | 88.95% | **83.05%** | 81.67% | 82.35% |
| Gaussian Naive Bayes | 88.95% | 80.95% | 85.00% | 82.93% |

### Final Observations

- **Logistic Regression** achieved the highest Accuracy.
- **Logistic Regression** achieved the highest Recall.
- **Logistic Regression** achieved the highest F1 Score.
- **KNN** achieved the highest Precision.
- **Gaussian Naive Bayes** achieved the second-highest Recall and F1 Score.
- Logistic Regression provided the strongest overall performance according to the selected evaluation criteria.

---

## 🏆 Best Model Selection

### Logistic Regression

Logistic Regression was selected as the **best overall model** after
feature engineering and hyperparameter tuning.

### Final Performance

| Metric | Score |
|---|---:|
| Accuracy | **90.00%** |
| Precision | 81.54% |
| Recall | **88.33%** |
| F1 Score | **84.80%** |

F1 Score was used as the primary metric for overall model selection
because it provides a balance between Precision and Recall.

Logistic Regression achieved the highest F1 Score (**84.80%**),
Accuracy (**90.00%**), and Recall (**88.33%**) among the final tuned
models.

KNN achieved the highest Precision (**83.05%**), showing that no single
model was best across every metric. However, Logistic Regression
provided the strongest overall balance between Precision and Recall.

Precision remains an important metric for this loan-approval problem
because false-positive approvals represent cases where the model
predicts a loan as Approved while the actual class is Rejected.

Based on the primary selection criterion and the overall evaluation,
Logistic Regression was selected as the final model.

### Final Confusion Matrix

![Final Confusion Matrix](images/confusion_matrix.png)

For the final Logistic Regression model:

```text
                 Predicted
                 Rejected  Approved

Actual Rejected     118       12
Actual Approved       7       53
```

This corresponds to:

- **True Negatives:** 118
- **False Positives:** 12
- **False Negatives:** 7
- **True Positives:** 53

---

## 💡 Key Takeaways

- Numerical and categorical features require different preprocessing strategies.
- Missing target values should be removed rather than imputed because their true class is unknown.
- A `ColumnTransformer` provides a clean way to apply different preprocessing operations to different feature types.
- Feature engineering improved the performance of all three evaluated models.
- KNN benefited the most from the engineered features in terms of F1 Score.
- Hyperparameter tuning improved the final Logistic Regression performance.
- Model selection should consider the business consequences of different types of classification errors.
- The model with the highest score on one metric is not necessarily the best model overall.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📁 Project Structure

```text
CreditWise-Loan-Approval/
│
├── images/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── model_comparison.png
│   └── target_distribution.png
│
├── credit_wise.ipynb
├── loan_approval_data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚠️ Disclaimer

This project is developed for **educational and portfolio purposes only**.

The model should not be used as a real-world automated financial decision-making system without further validation, fairness analysis, domain expertise, regulatory review, and testing on representative real-world data.