# PROJECT REPORT: CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
### Oasis Infobyte Data Analytics Internship (Project 3 - Level 2)
**Author:** Jasmin Jamadar  
**Role:** Data Analyst Intern  
**Submission Date:** June 2026

---

## 1. Executive Summary
This report presents a comparative machine learning study to build a Credit Card Fraud Detection System. Financial fraud accounts for billions of dollars in annual losses. Detecting fraudulent transactions in real-time is a core priority for banking institutions. However, the task is characterized by an extreme class imbalance, as fraudulent transactions represent a tiny fraction (under **0.17%**) of total volume.

Using a dataset of **284,807 transactions** (`creditcard.csv`), we built an NLP-style tabular preprocessing and modeling pipeline. Data cleaning involved the detection and deletion of **1,081 duplicate rows**, resulting in **283,726 unique transactions** containing PCA-transformed features (V1-V28), transaction Time, and Amount. The dataset was partitioned using a stratified 80% train / 20% test split. Continuous columns `Time` and `Amount` were scaled using `RobustScaler`. To prevent majority class bias, the training set was under-sampled, creating a balanced 50/50 training subset (756 samples).

Two classifiers were trained on this balanced data and evaluated on the imbalanced holdout test set (56,746 samples): **Logistic Regression** and **Random Forest Classifier**.
*   **Logistic Regression** achieved an Accuracy of **97.24%**, a Recall of **87.37%**, and an ROC-AUC of **95.54%**.
*   **Random Forest Classifier** achieved an Accuracy of **98.80%**, a Recall of **86.32%**, and an ROC-AUC of **96.97%**.

Crucially, **Random Forest Classifier** achieved a Precision of **10.96%** (flagging 666 false positives) compared to Logistic Regression's **5.08%** (flagging 1,624 false positives), cutting false alarms in half. Random Forest is the recommended model for deployment, providing substantial operational savings in analyst review time.

---

## 2. Introduction & Background
As credit card transactions have transitioned online, card-not-present fraud has escalated rapidly. Fraudsters use automated scripts, phishing, and identity theft to execute unauthorized transactions. Machine learning models must intercept these transactions in real-time before approval.

A key technical challenge is **extreme class imbalance**. Standard classifiers seeking to maximize accuracy will predict that every transaction is legitimate, achieving 99.83% accuracy but failing to catch any fraud. To build a viable detector, we must apply sampling techniques to force the model to learn the specific features of fraud.

---

## 3. Problem Statement & Objectives
The primary objectives of this project are:
1.  **Data Quality Control**: Load the dataset, verify columns, and remove duplicate rows to prevent evaluation metrics from being artificially inflated.
2.  **Exploratory Data Analysis (EDA)**: Map class distribution and analyze the transaction amount characteristics.
3.  **Outlier & Imbalance Preprocessing**: Scale continuous variables using robust estimators and resolve imbalance using Random Under-sampling on the training split.
4.  **Model Fitting**: Build and train **Logistic Regression** and **Random Forest Classifier** models.
5.  **Comparative Performance Check**: Evaluate models on accuracy, precision, recall, F1, and ROC-AUC metrics.
6.  **Prevention Guide**: Rank feature importances and outline actionable recommendations for payment gateways.

---

## 4. Dataset Description & Preprocessing Pipeline

### 4.1 Dataset Structure
The dataset used is `creditcard.csv` from Kaggle, representing transactions made by European cardholders in September 2013:
*   `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
*   `Amount`: Transaction amount.
*   `V1` to `V28`: Latent numerical variables resulting from a Principal Component Analysis (PCA) dimensionality reduction, designed to protect cardholder privacy.
*   `Class`: Target label (1 for fraud, 0 for legitimate).

### 4.2 Data Cleaning
The raw dataset contains **284,807 transactions**.
*   **Missing values**: 0 null entries were present.
*   **Duplicates**: **1,081 duplicate transactions** were identified and deleted, resulting in **283,726 unique transactions** (473 fraud, 283,253 legitimate).

### 4.3 Partitioning and Scaling
To ensure evaluation integrity, we split the unique records:
*   **Training Set (80%)**: 226,980 rows (378 fraud, 226,602 legitimate).
*   **Testing Set (20%)**: 56,746 rows (95 fraud, 56,651 legitimate).

Continuous columns `Time` and `Amount` exhibit high skew and extreme outliers. We applied `RobustScaler` to these columns:

$$x_{\text{scaled}} = \frac{x - \text{median}}{\text{IQR}}$$

RobustScaler is superior to StandardScaler because it uses the median and IQR, preventing extreme transaction amounts from distorting the feature bounds.

### 4.4 Resolving Class Imbalance (Random Under-sampling)
To balance the training data, we extracted all **378 fraud cases** from the training set and randomly selected **378 legitimate cases** from the same training split. This yielded a balanced training set of **756 observations**. Shuffling was applied to ensure stochastic gradients update uniformly. 

*Crucial Step:* Under-sampling was applied **only** to the training split. The test set remained untouched and imbalanced to ensure evaluation results accurately reflect real-world performance.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Class Imbalance
The transaction distribution is extremely skewed:
*   **Legitimate**: 283,253 (~99.8333%)
*   **Fraud**: 473 (~0.1667%)

Because the ratio is roughly 1 fraud per 600 transactions, a log-scale visualization (saved in [fraud_distribution.png](../Visualizations/fraud_distribution.png)) was required.

### 5.2 Transaction Amount Analysis
Plotting log-scaled transaction amount distributions (saved in [transaction_amount_distribution.png](../Visualizations/transaction_amount_distribution.png)) reveals distinct behavior:
*   **Legitimate Transactions** show a large spike near 1 currency unit (small transactions) and have a broad, right-skewed distribution stretching to 25,000 units.
*   **Fraudulent Transactions** are more concentrated, featuring a significant peak at 1 currency unit (potentially card verification tests) and another cluster around 100 units, indicating targeted theft sizes.

---

## 6. Model Training & Comparison Results

Both classifiers were trained on the balanced 756-row dataset and predicted on the holdout test set (56,746 samples):

### 6.1 Performance Evaluation Metrics Table

| Model Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 97.24% | 5.08% | **87.37%** | 9.60% | 95.54% | **0.022 seconds** |
| **Random Forest** | **98.80%** | **10.96%** | 86.32% | **19.45%** | **96.97%** | 0.163 seconds |

*Interpretation:* Random Forest outperforms Logistic Regression across almost all core metrics, particularly in F1-score and ROC-AUC.

### 6.2 The Precision-Recall Trade-off in Fraud
Due to the extreme imbalance, the F1-score is low for both models. This is driven by **Precision**:
*   **Logistic Regression**: Flags 1,624 legitimate transactions as fraud (False Positives), yielding a Precision of **5.08%**.
*   **Random Forest**: Flags only 666 legitimate transactions as fraud, yielding a Precision of **10.96%**.

While both models have near-identical coverage of actual frauds (87.37% vs. 86.32% Recall), **Random Forest reduces false alarms by 59%**. 

---

## 7. Model Diagnostics & Insights

### 7.1 Confusion Matrices Analysis
The confusion matrices (saved in [confusion_matrix.png](../Visualizations/confusion_matrix.png)) detail the following:
*   **Logistic Regression**: Out of 95 frauds, it catches 83 (12 missed). However, it flags 1,624 legitimate transactions as fraud.
*   **Random Forest**: Out of 95 frauds, it catches 82 (13 missed). However, it flags only 666 legitimate transactions as fraud.

### 7.2 Feature Importance (Random Forest)
The Gini feature importances plot (saved in [feature_importance.png](../Visualizations/feature_importance.png)) indicates that the top features are:
1.  **V14** (20.14% importance weight)
2.  **V10** (12.23%)
3.  **V17** (11.78%)
4.  **V12** (10.10%)
5.  **V3** (8.05%)

These four features account for over **54% of the decision weight**, indicating that these specific latent dimensions are highly sensitive to fraudulent transaction structures.

---

## 8. Actionable Recommendations for Payment Processors

1.  **Deploy Random Forest for Operational Efficiency**: By adopting the Random Forest Classifier over Logistic Regression, a financial institution can reduce the volume of false alerts by 59% while maintaining a high fraud capture rate (86%). This directly cuts down on call-center operational costs.
2.  **Implement a Tiered Risk Strategy**:
    *   **High Risk (Probability > 85%)**: Automatically decline the transaction and temporarily lock the card.
    *   **Medium Risk (Probability 50% to 85%)**: Allow the transaction but prompt the cardholder with a real-time SMS/App push for 2-Factor authentication.
    *   **Low Risk (Probability < 50%)**: Approve transaction immediately.
3.  **Real-Time Latent Feature Engineering**: Integrate the calculation of V14, V10, and V12 components directly into the real-time scoring API to evaluate transaction risk in milliseconds.

---

## 9. Conclusion & Future Improvements
We successfully developed a Credit Card Fraud Detection System. Random Forest emerged as the superior model, balancing fraud coverage (86.32% Recall) and false alarm mitigation (10.96% Precision, 96.97% AUC).

### Future Improvements
1.  **Synthetic Minority Over-sampling (SMOTE)**: Test SMOTE or ADASYN to synthetically generate fraudulent training points, rather than discarding non-fraud training data via under-sampling.
2.  **Advanced Boosting (XGBoost/LightGBM)**: Train gradient boosting models which frequently achieve higher AUC on highly imbalanced tabular datasets.
3.  **Unsupervised Anomaly Detection**: For novel or evolving fraud patterns, integrate unsupervised models like **Isolation Forest** or **One-Class SVM** which flag transactions based on deviation from standard buyer profiles.
