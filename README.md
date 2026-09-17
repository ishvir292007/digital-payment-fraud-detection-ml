# Digital Payment Fraud Detection ML

An end-to-end machine learning pipeline for detecting fraudulent digital payment transactions using synthetic transaction data, exploratory data analysis, feature engineering, SMOTE-based class balancing, and comparative classification models.

> **Project:** Paytm Fraud Detection
> **Domain:** FinTech / Digital Payments / Financial Crime Detection
> **Type:** Machine Learning Classification
> **Environment:** Python / Google Colab

---

## Overview

Digital payment platforms process millions of transactions, making automated fraud detection an important component of transaction security.

This project develops a machine learning pipeline designed to identify potentially fraudulent digital transactions before completion. The workflow covers the complete ML lifecycle, from synthetic data generation and data validation to exploratory analysis, preprocessing, class balancing, model training, and evaluation.

Because real financial transaction data is sensitive, this project uses a **synthetically generated dataset** that simulates transaction characteristics associated with legitimate and fraudulent activity.

---

## Problem Statement

Fraudulent digital transactions can result in:

* Financial losses
* Customer trust issues
* Increased operational costs
* Transaction security risks
* Reputational damage

The objective of this project is to build a machine learning classification pipeline capable of distinguishing between **legitimate** and **fraudulent** transactions based on transaction and behavioral features.

---

## Key Features

* Synthetic financial transaction dataset generation
* 200,000 transaction records
* 1% synthetic fraud rate
* Data-quality validation
* Missing-value analysis
* Duplicate detection
* Exploratory Data Analysis
* Transaction amount analysis
* Fraud distribution analysis
* Correlation analysis
* Behavioral fraud feature analysis
* Feature scaling using `StandardScaler`
* Class imbalance handling using **SMOTE**
* Multiple machine learning classifiers
* Confusion matrix analysis
* Accuracy, Precision, Recall and F1 evaluation
* ROC-AUC evaluation
* Comparative model assessment

---

## Dataset

The dataset is generated programmatically and contains **200,000 synthetic transactions**, including **2,000 fraudulent transactions**. The resulting class distribution is approximately **99% legitimate and 1% fraudulent**.

### Features

| Feature                    | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `transaction_amount`       | Monetary value of the transaction                    |
| `transaction_time_seconds` | Transaction time represented in seconds              |
| `failed_attempts`          | Number of failed transaction attempts                |
| `account_age_days`         | Age of the account in days                           |
| `location_distance_km`     | Distance associated with the transaction location    |
| `device_changes_30d`       | Number of device changes during the previous 30 days |
| `is_new_device`            | Indicates whether a new device was used              |
| `is_international`         | Indicates whether the transaction is international   |
| `ip_address_changed`       | Indicates whether the IP address changed             |
| `is_fraud`                 | Target variable: 0 = legitimate, 1 = fraudulent      |

The synthetic fraud-generation logic intentionally makes fraudulent transactions more distinctive through factors such as higher transaction amounts, increased failed attempts, new-device usage, international transactions, larger location distances, device changes, and IP changes.

---

## Machine Learning Pipeline

```text
Synthetic Transaction Generation
              │
              ▼
        Data Validation
              │
              ▼
     Exploratory Data Analysis
              │
              ▼
       Feature Preparation
              │
              ▼
        Train/Test Split
              │
              ▼
       Feature Scaling
              │
              ▼
            SMOTE
              │
              ▼
      Model Training
              │
       ┌──────┼──────────┐
       ▼      ▼          ▼
   Logistic  Decision   Random
  Regression   Tree      Forest
       │      │          │
       └──────┼──────────┘
              ▼
       Model Evaluation
              │
              ▼
   Fraud Detection Analysis
```

---

## Exploratory Data Analysis

The notebook performs several exploratory analyses to understand the transaction dataset and potential fraud patterns.

### Analysis includes

* Transaction amount distribution
* Legitimate vs fraudulent transaction distribution
* Transaction amount comparison by fraud status
* Feature correlation matrix
* New-device behavior
* International transaction behavior
* IP-address change behavior

These analyses are used to understand relationships within the synthetic dataset before model development.

---

## Data Quality

Before model development, the dataset is checked for:

* Missing values
* Duplicate records
* Appropriate data types

The generated dataset contains no missing values or duplicate rows in the executed notebook output.

---

## Models

The project uses the following classification algorithms:

### 1. Logistic Regression

A linear classification model used as a baseline for binary fraud classification.

### 2. Decision Tree Classifier

A tree-based model capable of learning non-linear decision boundaries and feature interactions.

### 3. Random Forest Classifier

An ensemble of decision trees used to model more complex relationships within transaction data.

The notebook imports and uses these scikit-learn classifiers as part of the model-building workflow.

---

## Handling Class Imbalance

Fraud datasets are typically highly imbalanced, meaning legitimate transactions substantially outnumber fraudulent transactions.

This project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address the class imbalance during model preparation.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

SMOTE is particularly relevant to this project because the synthetic dataset contains approximately **1% fraudulent transactions**.

---

## Evaluation Metrics

The models are evaluated using multiple classification metrics rather than relying solely on accuracy.

### Accuracy

Measures the overall proportion of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraudulent are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are successfully identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between fraudulent and legitimate transactions across classification thresholds.

### Confusion Matrix

Provides a breakdown of:

* True Positives
* True Negatives
* False Positives
* False Negatives

The notebook imports these evaluation metrics from scikit-learn.

---

## Technology Stack

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Core programming language       |
| Pandas           | Data manipulation               |
| NumPy            | Numerical computation           |
| Matplotlib       | Data visualization              |
| Seaborn          | Statistical visualization       |
| Scikit-learn     | Machine learning and evaluation |
| Imbalanced-learn | SMOTE-based class balancing     |
| Google Colab     | Development environment         |

---

## Project Structure

```text
digital-payment-fraud-detection-ml/
│
├── Untitled15 (2).ipynb
├── README.md
└── LICENSE
```

For a more production-oriented implementation, the project can later be organized as:

```text
digital-payment-fraud-detection-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── training.py
│   └── evaluation.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

Install Python 3.x and the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run with Google Colab

1. Open the notebook in Google Colab.
2. Run the library-import cells.
3. Execute the synthetic dataset generation.
4. Perform data validation and EDA.
5. Run preprocessing and class balancing.
6. Train the classification models.
7. Review the evaluation metrics and visualizations.

---

## Synthetic Data Disclaimer

This project **does not use real Paytm customer or transaction data**.

The dataset is synthetically generated for educational and machine-learning experimentation because real financial transaction data is sensitive and generally unavailable for unrestricted use.

Therefore, model performance demonstrated by this project should **not be interpreted as real-world Paytm fraud-detection performance**.

---

## Security & Privacy

No real customer information is used in this project.

The dataset is generated locally through a reproducible synthetic-data generation function, making the project suitable for educational experimentation without exposing real financial records.

---

## Future Improvements

Potential extensions include:

* Real-world anonymized transaction datasets
* Advanced feature engineering
* Transaction velocity features
* Time-window behavioral analysis
* User-level behavioral profiling
* Gradient Boosting models
* XGBoost / LightGBM experimentation
* Hyperparameter optimization
* Cross-validation
* Probability calibration
* Explainable AI using SHAP
* Real-time fraud scoring API
* Model monitoring
* Data drift detection
* Fraud-risk dashboards
* Dockerized deployment
* FastAPI inference service
* CI/CD integration

---

## Limitations

The primary limitation is the use of **synthetically generated data**.

The fraud labels are artificially created and the fraud-generation process deliberately introduces characteristics that make fraudulent transactions more distinguishable. Consequently, the resulting model performance may not represent performance on real-world financial transactions.

The project should therefore be considered a **machine-learning prototype and educational FinTech project**, rather than a production fraud-detection system.

---

## Learning Outcomes

This project demonstrates practical understanding of:

* Binary classification
* Fraud detection concepts
* Imbalanced machine learning
* Synthetic financial data generation
* Exploratory data analysis
* Feature preprocessing
* SMOTE
* Classification algorithms
* Model evaluation
* Confusion matrices
* ROC-AUC analysis
* FinTech risk analytics

---

## License

This project is intended for educational and research purposes.

Add an appropriate open-source license to the repository before distributing the project publicly.

---

## Author

**Ishvir Singh Matharoo**

BBA FinTech & AI
Chitkara University

---

## Project Status

**Status:** Academic / Educational ML Project

**Domain:** FinTech • Machine Learning • Fraud Detection • Digital Payments
