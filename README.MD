# CreditWise — Loan Approval Prediction

A Machine Learning classification project that predicts whether a loan application is **Approved** or **Rejected** using applicant, financial, employment, and credit-related information.

🔗 **[Live Demo](https://creditwise-loan-predict.streamlit.app/)**

---

## 📌 Overview

CreditWise is an end-to-end supervised Machine Learning classification project developed to predict loan approval outcomes from historical loan application data.

The project covers the complete workflow from data understanding and exploratory data analysis to preprocessing, model comparison using cross-validation, hyperparameter tuning, and final model selection.

Three classification algorithms were evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🎯 Problem Statement

Loan applications are evaluated using factors such as income, employment status, credit score, existing loans, debt-to-income ratio, savings, collateral, and loan requirements.

The objective of CreditWise is to build a Machine Learning system that can predict whether a loan application is likely to be **Approved** or **Rejected** based on historical application data.

Because incorrect approval and rejection decisions can have financial consequences, multiple evaluation metrics are considered rather than relying only on accuracy.

---

## 📊 Dataset

The original dataset contains:

- **1,000 loan applications**
- **20 columns**

Rows with missing values in the target variable `Loan_Approved` were removed because supervised learning requires a known target label.

After removing rows with missing target values, **950 labeled records** were available for model development.

### Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier (dropped before modeling) |
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

`Applicant_ID` is treated as an identifier and is not used as a predictive feature.

### Target Variable

`Loan_Approved`

```text
0 → Rejected
1 → Approved
```

---

## 🔎 Exploratory Data Analysis

### Loan Approval Distribution

![Loan Approval Distribution](images/target_distribution.png)

After removing records with missing target values, the labeled dataset contains:

- 650 Rejected applications
- 300 Approved applications

This corresponds to approximately:

- 68.42% Rejected
- 31.58% Approved

The target distribution was considered when interpreting model performance.

### Correlation Analysis

![Correlation Heatmap](images/correlation_heatmap.png)

Correlation analysis was performed on numerical features together with the target variable on the training set to understand relationships between the numerical variables and loan approval without data leakage.

Among the numerical features, Credit Score showed the strongest positive relationship with loan approval, while DTI Ratio showed the strongest negative relationship.

### Feature Distributions

![Feature Distributions](images/feature_distributions.png)

The distributions of the numerical variables were inspected to understand their spread, scale, and general behavior before model development.

---

## 🔄 Machine Learning Workflow

```text
Data Loading & Inspection
     ↓
Initial Exploratory Data Analysis
     ↓
Data Cleaning (Drop Missing Target & Applicant ID)
     ↓
Train-Test Split (80% Train, 20% Held-Out Test)
     ↓
Feature Type Definition (11 Numerical, 1 Ordinal, 6 Nominal)
     ↓
Preprocessing Pipeline (ColumnTransformer)
     ├── Numerical: SimpleImputer(mean) + StandardScaler
     ├── Ordinal: SimpleImputer(most_frequent) + OrdinalEncoder
     └── Nominal: SimpleImputer(most_frequent) + OneHotEncoder
     ↓
Baseline 5-Fold Cross-Validation on Train Data
     ├── Logistic Regression
     ├── KNN
     └── Gaussian Naive Bayes
     ↓
Hyperparameter Tuning (GridSearchCV with scoring="f1")
     ↓
Final Model Selection (Logistic Regression)
     ↓
Single Final Evaluation on Held-Out Test Set
     ↓
Pipeline Serialization (model.pkl) & Streamlit Deployment
```

---

## ⚙️ Data Preprocessing

To prevent data leakage, all preprocessing transformations are encapsulated within scikit-learn `Pipeline` and `ColumnTransformer` objects fitted exclusively on the training set.

### Numerical Features

Missing numerical values are imputed using mean imputation, followed by standardization using `StandardScaler`.

```text
Numerical Features (11)
       ↓
SimpleImputer(strategy="mean")
       ↓
StandardScaler()
```

Scaling is critical for distance-sensitive models like KNN and helps gradient-based solvers in Logistic Regression converge efficiently.

### Categorical Features

Categorical features are divided into ordinal and nominal subsets:

#### Education Level (Ordinal)

`Education_Level` contains two categories with a natural educational progression:
- Not Graduate
- Graduate

These are encoded using `OrdinalEncoder`:

```text
Not Graduate → 0
Graduate     → 1
```

#### Other Categorical Features (Nominal)

The remaining six categorical variables (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`) are treated as nominal features:

```text
Categorical Features (6)
       ↓
SimpleImputer(strategy="most_frequent")
       ↓
OneHotEncoder(drop="first", handle_unknown="ignore")
```

- `drop="first"` removes collinear reference categories.
- `handle_unknown="ignore"` ensures that new or unseen categories at inference time are handled gracefully without raising errors.

### Target Encoding

The target variable is binary encoded before model training:

```text
No  → 0
Yes → 1
```

---

## 🤖 Models Evaluated

### 1. Logistic Regression

Logistic Regression serves as a linear classification algorithm predicting loan approval probability via the log-odds function. It provides highly interpretable coefficients and well-calibrated probabilities, making it the industry standard for credit scoring.

### 2. K-Nearest Neighbors (KNN)

KNN classifies applicants based on the majority label among their $k$ nearest neighbors in the standardized feature space.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier applying Bayes' Theorem under the assumption of feature conditional independence.

---

## 📏 Evaluation Metrics

The models are evaluated using:

- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Proportion of predicted approvals that are actual approvals ($rac{TP}{TP + FP}$). High precision minimizes costly false approvals.
- **Recall**: Proportion of actual approvals correctly approved by the model ($rac{TP}{TP + FN}$).
- **F1 Score**: Harmonic mean of Precision and Recall ($rac{2 \cdot P \cdot R}{P + R}$), providing the primary balanced selection metric.
- **Confusion Matrix**: Granular count of True Negatives, False Positives, False Negatives, and True Positives.

---

## 🧪 Baseline Cross-Validation Results

The three core models were evaluated using 5-fold stratified cross-validation on the training set ($N = 760$):

| Model | CV F1 (Mean) | CV F1 (Std) | CV Accuracy (Mean) |
|---|---:|---:|---:|
| **Logistic Regression** | **77.28%** | ± 3.09% | 86.18% |
| **KNN** | 73.01% | ± 2.93% | 83.42% |
| **Gaussian Naive Bayes** | **77.51%** | ± 5.48% | 86.97% |

### Baseline Observations

- **Logistic Regression** and **Gaussian Naive Bayes** achieved comparable baseline CV F1 scores (~77.3% vs ~77.5%).
- Logistic Regression demonstrated greater stability across folds, with a standard deviation of only **3.09%** compared to **5.48%** for Naive Bayes.
- **KNN** trailed both probabilistic models, achieving 73.01% CV F1.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was conducted using `GridSearchCV` with 5-fold cross-validation on the training data.

In accordance with project objectives, the tuning optimization objective was:

```python
scoring = "f1"
```

### Tuning Grids & Best Parameters

1. **Logistic Regression**:
   - Parameter Grid: `C = [0.01, 0.1, 1, 10, 100]`
   - **Best Parameter**: `C = 1`
   - **Best 5-Fold CV F1**: **77.28%**

2. **K-Nearest Neighbors (KNN)**:
   - Parameter Grid: `n_neighbors = [3, 5, 7, 9, 11, 15]`
   - **Best Parameter**: `n_neighbors = 15`
   - **Best 5-Fold CV F1**: **74.75%**

3. **Gaussian Naive Bayes**:
   - Parameter Grid: `var_smoothing = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]`
   - **Best Parameter**: `var_smoothing = 1e-11`
   - **Best 5-Fold CV F1**: **77.51%**

### Best Hyperparameters Summary

| Model | Best Hyperparameter | Best CV F1 Score |
|---|---|---:|
| **Logistic Regression** | `C = 1` | **77.28%** |
| **KNN** | `n_neighbors = 15` | 74.75% |
| **Gaussian Naive Bayes** | `var_smoothing = 1e-11` | **77.51%** |

---

## 🏆 Final Model Selection

### Selected Model: Logistic Regression (`C = 1`)

**Logistic Regression** was selected as the final production model for three key reasons:

1. **Performance & Stability**: Achieved a competitive 5-fold CV F1 score of **77.28%** with low fold variance ($\sigma = 0.0309$).
2. **Calibrated Probability Outputs**: Unlike Naive Bayes (whose conditional independence assumption skews predicted probabilities toward 0 or 1), Logistic Regression produces calibrated probabilities suitable for risk thresholding and tiered underwriting.
3. **Interpretability**: Linear model coefficients directly represent feature contributions (odds ratios), providing clear transparency into how individual applicant features influence loan approval predictions.

---

## 📊 Final Held-Out Test Set Evaluation

Following strict train/CV/test discipline, the finalized Logistic Regression pipeline was evaluated **once** on the completely held-out test set ($N = 190$):

| Metric | Score |
|---|---:|
| **Accuracy** | **88.95%** (169 / 190) |
| **Precision** | **85.45%** |
| **Recall** | **78.33%** |
| **F1 Score** | **81.74%** |

### Confusion Matrix

```text
                 Predicted
                Rejected  Approved

Actual Rejected    122        8
Actual Approved     13       47
```

- **True Negatives**: **122** rejected applications correctly classified as rejected.
- **False Positives**: **8** rejected applications incorrectly classified as approved (low rate, protecting the lender from bad credit risk).
- **False Negatives**: **13** approved applications incorrectly classified as rejected.
- **True Positives**: **47** approved applications correctly identified.

---

## 💡 Key Takeaways

- **Leak-Free Preprocessing**: Encapsulating numerical scaling, missing value imputation, and categorical encoding inside a single `ColumnTransformer` within an sklearn `Pipeline` prevents data leakage and ensures reproducible inference.
- **F1-Optimized Tuning**: Tuning hyperparameters with `scoring="f1"` properly balanced precision and recall rather than biasing predictions toward the majority rejected class.
- **Model Parsimony**: The original 18 clean, domain-meaningful features provide strong predictive performance without introducing structural multicollinearity or complex feature dependencies.
- **Calibrated Decision Making**: Logistic Regression provides the probability calibration and explainability essential for responsible automated lending workflows.

---

## 🛠️ Technologies Used

- **Python 3.13**
- **Scikit-learn** (Pipelines, ColumnTransformer, GridSearchCV, Metrics)
- **Pandas** & **NumPy** (Data processing)
- **Matplotlib** & **Seaborn** (Exploratory data visualization)
- **Streamlit** (Interactive web application deployment)
- **Joblib** (Model serialization)

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
├── credit_wise.ipynb          # Clean, leak-free Jupyter notebook workflow
├── app.py                     # Streamlit web application
├── model.pkl                  # Serialized final scikit-learn pipeline
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
└── .gitignore
```

---

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes only.

The model is not intended to be used as a real-world automated financial decision-making system. A production loan approval system requires substantially larger datasets, fairness and bias audits, stress testing across economic cycles, and human oversight.

---

## 🖥️ Streamlit Demo

🔗 **[Live Demo](https://creditwise-loan-predict.streamlit.app/)**

CreditWise includes an interactive Streamlit web application that accepts applicant, employment, financial, and loan information and generates predictions and approval probabilities using the trained Logistic Regression pipeline.

The application includes:

- Interactive inputs for applicant, employment, financial, and loan details
- Input bounds validation (preventing negative income, loans, etc.)
- Out-of-distribution warnings for inputs outside observed training ranges
- Instant loan approval prediction (**Approved** vs **Rejected**)
- Estimated approval probability display

### Running the Application Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Access the web interface in your browser at `http://localhost:8501`.
