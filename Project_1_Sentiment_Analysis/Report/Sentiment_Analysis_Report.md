# PROJECT REPORT: SENTIMENT ANALYSIS SYSTEM FOR TWITTER DATA
### Oasis Infobyte Data Analytics Internship (Project 1 - Level 1)
**Author:** Jasmin  
**Role:** Data Analytics Intern  
**Submission Date:** June 2026

---

## 1. Executive Summary
This project presents the design, implementation, and evaluation of a machine learning-based Sentiment Analysis System built using Python. In the modern digital era, understanding public sentiment from unstructured text data like social media posts (tweets) is crucial for organizations, governments, and businesses to gauge brand reputation, political trends, and customer satisfaction.

Using a dataset of over 160,000 tweets (`Twitter_Data.csv`), a complete Natural Language Processing (NLP) pipeline was developed. After extensive text cleaning and TF-IDF feature extraction, two classifiers were trained: **Logistic Regression** and **Multinomial Naive Bayes**. 

The results demonstrate that **Logistic Regression** achieved an overall accuracy of **89.04%** and a macro F1-score of **87.88%**, significantly outperforming the **Multinomial Naive Bayes** baseline, which achieved **72.92%** accuracy and **70.15%** F1-score. This report provides a detailed breakdown of the methodology, visualization of the data distributions, analysis of key sentiments using word clouds, model performance evaluation, and recommendations for future work.

---

## 2. Introduction & Objective
Social media platforms generate millions of short text posts every day. Extracting actionable insights from these text streams is a primary challenge in Natural Language Processing (NLP) due to the presence of slang, emojis, URLs, typos, and irregular grammar.

The main objective of this project is to build an automated sentiment classifier that:
1.  Ingests a corpus of raw tweets.
2.  Applies advanced preprocessing techniques to clean and standardize text data.
3.  Performs exploratory data analysis (EDA) to study class imbalances.
4.  Transforms raw text into sparse TF-IDF vectors.
5.  Trains, compares, and evaluates **Logistic Regression** and **Multinomial Naive Bayes** classifiers on accuracy, precision, recall, and F1-score.
6.  Generates confusion matrices to diagnose classification errors.
7.  Extracts key textual indicators of Positive, Neutral, and Negative sentiments.

---

## 3. Dataset & Exploratory Data Analysis (EDA)

### 3.1 Dataset Overview
The dataset used for this project is `Twitter_Data.csv`, which consists of two columns:
*   `clean_text`: The raw text of the tweet.
*   `category`: Numerical sentiment labels:
    *   `1`: Positive
    *   `0`: Neutral
    *   `-1`: Negative

The dataset originally contained **162,980 records**. After dropping null rows (4 missing in `clean_text` and 7 in `category`), **162,969 records** were retained.

### 3.2 Sentiment Distribution
The distribution of sentiments across the cleaned dataset is summarized below:

*   **Positive (1)**: 72,249 samples (~44.4%)
*   **Neutral (0)**: 55,211 samples (~33.9%)
*   **Negative (-1)**: 35,509 samples (~21.8%)

The distribution indicates a mild imbalance, where positive sentiments represent nearly double the volume of negative sentiments. This imbalance was handled during modeling by utilizing a stratified train-test split to ensure representative sentiment ratios in both sets.

The visual representation of this distribution is saved in [sentiment_distribution.png](../Visualizations/sentiment_distribution.png).

---

## 4. Preprocessing & Feature Engineering Methodology
Raw text data contains noise that can degrade model performance. A structured pre-processing pipeline was constructed:

### 4.1 Data Cleaning Steps
1.  **Lowercasing**: All characters were converted to lowercase to ensure that words like "Good" and "good" map to the same vector index.
2.  **URL and HTML Tag Removal**: Web addresses (`http`, `https`, `www.`) and HTML tags were stripped using regular expressions, as they do not indicate sentiment.
3.  **Special Characters & Punctuation Removal**: Numbers, punctuation marks, and non-alphabetic symbols were replaced with whitespace to isolate core linguistic roots.
4.  **Tokenization**: The cleaned sentences were split into individual words.
5.  **Stopwords Removal**: Common grammatical words (e.g., 'the', 'is', 'for', 'about') which appear frequently but carry no sentimental charge were filtered out. Words shorter than 3 letters were also discarded.

*Example Preprocessing Output:*
*   **Raw Text:** `when modi promised minimum government maximum governance expected him begin the difficult job reforming the state why does take years get justice state should and not business and should exit psus and temples`
*   **Cleaned Text:** `modi promised minimum government maximum governance expected begin difficult job reforming state take years get justice state business exit psus temples`

### 4.2 TF-IDF Feature Extraction
The preprocessed text columns were converted to numerical matrices using Term Frequency-Inverse Document Frequency (TF-IDF). TF-IDF measures the importance of a term relative to a single document and the entire corpus:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

For this project, the `TfidfVectorizer` from `scikit-learn` was configured with:
*   `max_features=25000`: Selecting the top 25,000 most significant unigrams and bigrams.
*   `ngram_range=(1, 2)`: Capturing both single words (e.g., "good") and adjacent word pairs (e.g., "not good"), which helps preserve local context and negatives.

---

## 5. Model Training & Evaluation Results
The dataset was split into an **80% training set** (130,314 samples) and a **20% testing set** (32,579 samples), maintaining class balance via stratification.

### 5.1 Model Descriptions
1.  **Logistic Regression**: A linear model that estimates probabilities using a logistic function. The `lbfgs` solver was chosen to natively handle multiclass classification (multinomial loss). Regularization parameter $C=1.0$ was applied.
2.  **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' Theorem, assuming conditional independence between features. It is standard for text classification due to high efficiency. Laplace smoothing ($\alpha=1.0$) was enabled.

### 5.2 Performance Metrics Comparison
The evaluation of both models on the test set yielded the following metrics:

| Performance Metric | Logistic Regression | Multinomial Naive Bayes |
| :--- | :---: | :---: |
| **Accuracy** | **89.04%** | 72.92% |
| **Precision (Macro)** | **88.74%** | 77.09% |
| **Recall (Macro)** | **87.52%** | 68.09% |
| **F1-Score (Macro)** | **87.88%** | 70.15% |
| **Training Time (sec)**| 5.04 seconds | **0.02 seconds** |

The comparative bar chart visual is saved in [model_comparison.png](../Visualizations/model_comparison.png).

### 5.3 Detailed Classification Reports

#### Logistic Regression
```text
              precision    recall  f1-score   support

    Negative       0.88      0.76      0.82      7102
     Neutral       0.86      0.97      0.91     11028
    Positive       0.92      0.89      0.91     14449

    accuracy                           0.89     32579
   macro avg       0.89      0.88      0.88     32579
weighted avg       0.89      0.89      0.89     32579
```

#### Multinomial Naive Bayes
```text
              precision    recall  f1-score   support

    Negative       0.82      0.48      0.60      7102
     Neutral       0.82      0.66      0.73     11028
    Positive       0.67      0.91      0.77     14449

    accuracy                           0.73     32579
   macro avg       0.77      0.68      0.70     32579
weighted avg       0.75      0.73      0.72     32579
```

---

## 6. Discussion & Sentiment Insights

### 6.1 Analysis of Confusion Matrices
The confusion matrices (saved in [confusion_matrix.png](../Visualizations/confusion_matrix.png)) provide deep insights:
*   **Logistic Regression** has a high true positive rate across all classes. Its highest confusion is misclassifying 23.4% of Negative tweets as Neutral, which is a common challenge in sentiment analysis due to sarcastic or mild language.
*   **Multinomial Naive Bayes** struggles severely with Negative tweets, misclassifying 51.6% of them (3,663 out of 7,102) as Positive or Neutral. This is a common weakness of Naive Bayes when classes are imbalanced and feature independence assumptions are violated by short, context-dependent text. However, Naive Bayes achieved a high Recall of 91.1% on Positive tweets, suggesting it is highly biased towards predicting the majority positive class.

### 6.2 Key Drivers of Sentiment (Word Clouds)
By visual inspection of the generated word clouds (saved as [wordcloud_positive.png](../Visualizations/wordcloud_positive.png) and [wordcloud_negative.png](../Visualizations/wordcloud_negative.png)), we draw the following insights:
*   **Positive Sentiments**: Dominated by words of approval, support, and hope: `great`, `support`, `good`, `welcome`, `clean`, `development`, `growth`. The name `modi` and word `vote` are central, indicating a major political campaign focus.
*   **Negative Sentiments**: Dominated by words of disappointment, frustration, and critique: `nonsense`, `drama`, `corruption`, `state`, `fail`, `lies`, `media`. The term `modi` is also present, which demonstrates that the name is a focal point of both praise and criticism in political discourse.

---

## 7. Internship Learnings & Conclusion
Building this project yielded several critical learnings in the domain of Data Analytics and Natural Language Processing:
1.  **Preprocessing Impact**: Raw text is highly chaotic. Developing a robust, regular expression-based cleaning pipeline is the foundation of high-performance NLP models.
2.  **Imbalance Challenges**: Even a moderate class imbalance (44% Positive vs 22% Negative) can significantly bias simpler models like Naive Bayes towards the majority class.
3.  **Model Trade-offs**: Logistic Regression requires more training time (~5s) compared to Naive Bayes (~0.02s) but offers vastly superior predictive accuracy (89% vs 73%), making it the superior choice for production applications where classification accuracy is critical.
4.  **Feature Representation**: Utilizing bigrams in the TF-IDF representation was critical in capturing local context (such as "not good" or "no corruption"), which directly improved the classification of complex, negative reviews.

---

## 8. Recommendations & Future Scope
To further improve this sentiment analysis system, the following approaches are recommended:
1.  **VADER Sentiment Integration**: For social media text, combining machine learning with lexicon-based tools like VADER (Valence Aware Dictionary and sEntiment Reasoner) can improve accuracy on slang and emojis.
2.  **Class Balancing Techniques**: Applying SMOTE (Synthetic Minority Over-sampling Technique) or random undersampling on the training set could help Naive Bayes perform better on the minority Negative class.
3.  **Deep Learning / Transformer Models**: Implementing pre-trained transformer models such as BERT (Bidirectional Encoder Representations from Transformers) or RoBERTa would capture long-range contextual semantic structures and likely push accuracy past 95%.
4.  **Sarcasm Detection**: Incorporating feature-engineering strategies specifically for sarcasm detection (like punctuation density or exclamation analysis) would mitigate confusion between negative and neutral classes.
