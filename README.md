# Oasis Infobyte Data Analytics Internship Portfolio (OIBSIP)

### Completed Projects Showcase

**Intern:** Jasmin Jamadar\
**Role:** Data Analyst Intern\
**Internship Period:** June 2026

---

## Repository Overview

Welcome to my **Oasis Infobyte (OIBSIP) Data Analytics Internship** repository!
This portfolio showcases a series of end-to-end data analytics and machine
learning projects developed during my internship. The projects range from
Natural Language Processing (NLP) to predictive regression modeling and
class-imbalanced classification pipelines.

Each project is fully documented, structured, and recruiter-ready, featuring:

- Interactive **Jupyter Notebooks** with detailed markdown explanations.
- **Formal Internship Reports** summarizing data preprocessing, analytical
  findings, and recommendations.
- High-resolution **Visualizations** generated using Matplotlib and Seaborn.
- Production-ready, well-formatted python scripts.

---

## Project Portfolio Summary

Below is a summary of the three core projects completed under the internship:

| #     | Project Title                   | Key Objective                                                                       | Core Algorithms & Tech                                                     |          Key Performance Metric           |                    Project Folder                     |
| :---- | :------------------------------ | :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------- | :---------------------------------------: | :---------------------------------------------------: |
| **1** | **Sentiment Analysis**          | Classify movie/tweet reviews into Positive, Neutral, or Negative sentiments.        | TF-IDF, Logistic Regression, Multinomial Naive Bayes, NLTK, WordCloud      | **89.04% Accuracy** (Logistic Regression) |   [Go to Project 1](./Project_1_Sentiment_Analysis)   |
| **2** | **House Price Prediction**      | Estimate residential property valuations based on structural and location features. | Multiple Linear Regression, IQR outlier removal, MinMaxScaler, Statsmodels | **74.52% $R^2$ Score** (Goodness-of-Fit)  | [Go to Project 2](./Project_2_House_Price_Prediction) |
| **3** | **Credit Card Fraud Detection** | Identify fraudulent transactions under extreme class imbalance (0.17% fraud rate).  | Random Under-sampling, RobustScaler, Random Forest, Logistic Regression    |  **96.97% ROC-AUC** / **86.32% Recall**   |    [Go to Project 3](./Project_3_Fraud_Detection)     |

---

## Technology Stack & Libraries Used

The entire portfolio is implemented in **Python** using the following scientific
libraries:

- **Data Manipulation:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-Learn`
- **Natural Language Processing (NLP):** `NLTK`, `WordCloud`
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Statistical Analysis:** `Statsmodels`

---

## Global Repository Structure

```directory
OIBSIP_Jasmin (Workspace Root)
│
├── Project_1_Sentiment_Analysis
│   ├── Dataset (Twitter_Data.csv)
│   ├── Notebook (Sentiment_Analysis.ipynb)
│   ├── Report (Sentiment_Analysis_Report.md)
│   ├── Visualizations (*.png)
│   └── README.md
│
├── Project_2_House_Price_Prediction
│   ├── Dataset (Housing.csv)
│   ├── Notebook (House_Price_Prediction.ipynb)
│   ├── Report (House_Price_Prediction_Report.md)
│   ├── Visualizations (*.png)
│   └── README.md
│
├── Project_3_Fraud_Detection
│   ├── Dataset (creditcard.csv)
│   ├── Notebook (Fraud_Detection.ipynb)
│   ├── Report (Fraud_Detection_Report.md)
│   ├── Visualizations (*.png)
│   └── README.md
│
└── README.md (This global landing page)
```

---

## Installation & Execution Guide

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/OIBSIP_Jasmin.git
cd OIBSIP_Jasmin
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Install all necessary scientific and
machine learning libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
```

### 3. Run the Jupyter Environment

Launch Jupyter Notebook to interactively run the analysis pipelines:

```bash
jupyter notebook
```

---

## Global Internship Highlights & Key Learnings

Through this internship program with Oasis Infobyte, I successfully mastered
several core competencies in Data Science and NLP:

1. **Imbalance Solutions (Project 3)**: Implemented under-sampling techniques on
   a 284k-transaction credit card dataset, demonstrating how to address severe
   class imbalance and evaluate models using business-critical metrics like
   **Recall** and **ROC-AUC** rather than simple accuracy.
2. **Regression Diagnostics (Project 2)**: Practiced treating outliers via IQR
   filtering and scaling numeric columns without test data leakage, validating
   OLS assumptions (normality, homoscedasticity) to predict property values with
   a 74.5% explained variance.
3. **NLP Pipelines (Project 1)**: Built a text tokenization, punctuation/URL
   cleaning, and stopword-removal pipeline using NLTK, converting unstructured
   text into dense TF-IDF matrices to train highly accurate classification
   models.
4. **Portfolio Documentation**: Practiced writing formal machine learning
   reports and detailed READMEs to communicate complex analytic results clearly
   to business stakeholders and technical recruiters.

---

## Acknowledgment

I would like to express my sincere gratitude to **Oasis Infobyte** for offering
this structured internship and providing real-world datasets that helped bridge
the gap between academic learning and industry standards.
