# PROJECT REPORT: HOUSE PRICE PREDICTION USING MULTIPLE LINEAR REGRESSION

### Oasis Infobyte Data Analytics Internship (Project 2 - Level 2)

**Author:** Jasmin Jamadar **Role:** Data Analyst Intern **Submission Date:**
June 2026

---

## 1. Executive Summary

This report presents a structured analysis and machine learning implementation
to predict residential house prices in the Delhi region. Real estate valuations
are traditionally complex, driven by a combination of physical dimensions,
structural specifications, available amenities, and local market trends.

Using a dataset of **545 residential properties** (`Housing.csv`), we built an
end-to-end Multiple Linear Regression pipeline. Data cleaning involved the
identification and removal of **25 outliers** in property price and area using
the Interquartile Range (IQR) method. Binary features were mapped to numeric
formats, and categorical variables (such as furnishing status) were
dummy-encoded to prevent multi-collinearity. Continuous numerical columns were
normalized using a `MinMaxScaler`.

The trained Multiple Linear Regression model achieved a **Coefficient of
Determination ($R^2$) of 74.52%** on the hold-out test set, indicating that
approximately 74.5% of property value variation is explained by the features.
The **Mean Absolute Error (MAE)** was **630,394.79** and the **Root Mean Squared
Error (RMSE)** was **834,574.81**. The report highlights that **property area,
number of bathrooms, and stories** are the most critical determinants of house
prices, providing actionable investment strategies for developers and real
estate investors.

---

## 2. Introduction & Background

Predicting real estate prices is a classic regression problem in machine
learning. Valuation models serve as a foundation for real estate search engines,
tax assessments, bank loan approvals, and real estate investment trusts (REITs).
Accurately identifying how specific features impact price allows buyers and
sellers to make data-driven decisions.

Linear Regression is a fundamental statistical method used to model the
relationship between a dependent target variable (house price) and one or more
independent predictor features. OLS (Ordinary Least Squares) linear regression
seeks to find the line of best fit by minimizing the sum of squared differences
between actual and predicted targets.

---

## 3. Problem Statement & Objectives

The primary objectives of this project are:

1. **Data Preprocessing and Quality Control**: Load the dataset, handle missing
   data, check for duplicate rows, and systematically clean up leverage points
   (outliers) that bias linear regression lines.
2. **Exploratory Data Analysis (EDA)**: Conduct statistical summaries and
   correlation heatmaps to explore relationships between predictors and pricing.
3. **Feature Engineering**: Apply binary encoding and dummy variable encoding
   (while avoiding the dummy variable trap). Scale continuous features using
   min-max normalization.
4. **Model Building**: Train a multiple linear regression model using
   `scikit-learn`.
5. **Model Diagnostics**: Validate linear model assumptions by analyzing
   residuals (normality of errors and homoscedasticity).
6. **Business Recommendations**: Extract regression coefficients to rank feature
   importance and recommend investments for homeowners and builders.

---

## 4. Dataset Description & Preprocessing Pipeline

### 4.1 Dataset Characteristics

The dataset `Housing.csv` contains **545 observations** and **13 variables**:

- **Continuous numerical variables**: `price` (Target), `area` (in sq ft),
  `bedrooms`, `bathrooms`, `stories`, `parking` (number of vehicles).
- **Categorical text variables (Binary)**: `mainroad` (faces main road),
  `guestroom` (has guest room), `basement` (has basement), `hotwaterheating`
  (has hot water), `airconditioning` (has air conditioning), `prefarea`
  (preferred neighborhood).
- **Categorical text variables (Multi-class)**: `furnishingstatus` (Furnished,
  Semi-furnished, Unfurnished).

The raw dataset contained **0 missing values** and **0 duplicate records**,
indicating excellent data collection quality.

### 4.2 Outlier Analysis and Treatment

Linear regression coefficients are highly sensitive to outliers. Outlier
detection was conducted on the two continuous features (`price` and `area`)
using the Interquartile Range (IQR) method:

$$\text{IQR} = Q3 - Q1$$

$$\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}$$

$$\text{Upper Bound} = Q3 + 1.5 \times \text{IQR}$$

- **Price Outliers**: $Q1 = 3,430,000$, $Q3 = 5,740,000$, yielding bounds of
  $[35,000, 9,205,000]$. Houses priced above $9.205$ Million (15 records) were
  flagged.
- **Area Outliers**: $Q1 = 3,600$, $Q3 = 6,360$, yielding bounds of
  $[-540, 10,500]$. Properties with areas exceeding $10,500$ sq ft (12 records)
  were flagged.

A total of **25 rows** were dropped because they fell outside these bounds,
retaining **520 clean records** for training. Outlier removal corrected the
skewness in price distribution, enabling a more generalizable OLS fit.

### 4.3 Feature Encoding

To process text columns:

1. **Binary Mapping**: Columns `mainroad`, `guestroom`, `basement`,
   `hotwaterheating`, `airconditioning`, and `prefarea` were mapped to numeric
   formats using: $$\text{yes} \rightarrow 1, \quad \text{no} \rightarrow 0$$
2. **Multiclass Encoding**: Column `furnishingstatus` was encoded using One-Hot
   Dummy Variables. To avoid multicollinearity (the "dummy variable trap"), the
   first dummy feature was dropped, resulting in two features:
   `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished` (with
   `furnished` as the reference baseline).

### 4.4 Feature Scaling

Continuous features (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`) have
widely differing magnitudes (e.g. area in thousands, bedrooms in units). To
allow direct comparison of coefficients (feature importances), we scaled
features using `MinMaxScaler`, mapping values between $[0, 1]$:

$$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$

The scaler was fitted strictly on the **training dataset** (80% split) and
applied to both the training and test sets (20% split) to prevent data leakage.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Continuous Variables Statistical Summary (Cleaned Data)

- **Average price**: 4,491,152 (range: 1,750,000 to 9,100,000).
- **Average area**: 4,960 sq ft (range: 1,650 to 10,500 sq ft).
- **Average bedrooms**: 2.94 rooms.
- **Average bathrooms**: 1.25 rooms.
- **Average stories**: 1.77 floors.
- **Average parking spaces**: 0.65 spaces.

The summary indicates that the typical property in the dataset is a 3-bedroom,
1-bathroom home with an area of about 5,000 sq ft and less than 1 parking spot.

### 5.2 Correlation Heatmap Summary

A Pearson correlation matrix (saved in
[correlation_heatmap.png](../Visualizations/correlation_heatmap.png)) reveals:

- **Area** has the strongest positive correlation with **Price** ($r = 0.53$),
  followed by **Bathrooms** ($r = 0.49$) and **Stories** ($r = 0.43$).
- **Air Conditioning** shows a solid correlation with Price ($r = 0.46$).
- Categorical feature **prefarea** is positively correlated with Price
  ($r = 0.31$).
- **Unfurnished status** is negatively correlated with Price ($r = -0.28$),
  confirming that buyers pay a premium for furnished units.

---

## 6. Model Building & Coefficient Analysis

A Multiple Linear Regression model was fitted on the scaled training features.
The trained model takes the form:

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n$$

The intercept ($\beta_0$) was calculated as **$2,333,088.16$**, representing the
baseline price of an unfurnished, 1-story house with 0 area, 0 bedrooms, 0
bathrooms, and no amenities, located away from the main road in a non-preferred
area.

### 6.1 Regression Coefficients (Feature Impact)

Since continuous variables were scaled to $[0, 1]$, the regression coefficients
represent the change in house price when a feature transitions from its minimum
to its maximum value, holding all other features constant:

| Feature Name                        | Coefficient ($\beta$) | Absolute Impact | Direction |
| :---------------------------------- | :-------------------: | :-------------: | :-------- |
| **area**                            |   **+2,328,450.41**   | **23.28 Lakhs** | Positive  |
| **bathrooms**                       |   **+1,432,335.34**   | **14.32 Lakhs** | Positive  |
| **stories**                         |   **+1,400,942.33**   | **14.01 Lakhs** | Positive  |
| **hotwaterheating**                 |    **+761,858.91**    | **7.62 Lakhs**  | Positive  |
| **airconditioning**                 |    **+696,892.65**    | **6.97 Lakhs**  | Positive  |
| **prefarea**                        |    **+524,877.72**    | **5.25 Lakhs**  | Positive  |
| **parking**                         |    **+499,995.53**    | **5.00 Lakhs**  | Positive  |
| **furnishingstatus_unfurnished**    |    **-427,502.91**    | **4.28 Lakhs**  | Negative  |
| **basement**                        |    **+374,432.53**    | **3.74 Lakhs**  | Positive  |
| **mainroad**                        |    **+363,263.02**    | **3.63 Lakhs**  | Positive  |
| **bedrooms**                        |    **+304,887.65**    | **3.05 Lakhs**  | Positive  |
| **guestroom**                       |    **+258,205.12**    | **2.58 Lakhs**  | Positive  |
| **furnishingstatus_semi-furnished** |    **+38,514.40**     | **0.39 Lakhs**  | Positive  |

_Analysis:_ Property area, bathrooms, and stories dominate valuation weights.
Features like `bedrooms` and `guestrooms` have surprisingly low coefficients,
likely because their variance is already captured by overall area and stories.

---

## 7. Experimental Results & Performance Evaluation

The model was validated on the hold-out test set (104 samples). The evaluation
metrics are summarized below:

- **R-squared ($R^2$) Score**: **0.7452**\
  _Interpretation:_ 74.52% of the price variation in the test set is explained
  by the model, indicating a high-quality regression fit.
- **Mean Absolute Error (MAE)**: **630,394.79**\
  _Interpretation:_ The model's predictions differ from the actual house prices
  by an average of 6.30 Lakhs.
- **Mean Squared Error (MSE)**: **696,515,114,846.24**
- **Root Mean Squared Error (RMSE)**: **834,574.81**\
  _Interpretation:_ The standard deviation of the residuals is 8.35 Lakhs,
  representing the typical prediction error.

The prediction scatter plot (saved in
[actual_vs_predicted.png](../Visualizations/actual_vs_predicted.png)) shows a
tight clustering around the $y = x$ prediction line.

---

## 8. Model Diagnostics & Validation of Assumptions

To verify that Multiple Linear Regression was appropriate, we performed
diagnostic checks on the residuals ($e_i = y_i - \hat{y}_i$):

1. **Normality of Errors**: The distribution of residuals (visualized in the
   Jupyter Notebook) shows a symmetric, bell-shaped distribution centered
   around 0. This confirms that the assumption of normally distributed error
   terms holds.
2. **Homoscedasticity (Constant Variance)**: The residual plot (saved in
   [residual_plot.png](../Visualizations/residual_plot.png)) plots residuals
   against predicted prices. The residuals are randomly scattered within a
   horizontal band around the 0-line without forming a funnel shape. This
   confirms the constant variance assumption is satisfied.

---

## 9. Findings & Actionable Recommendations

### 9.1 Core Market Findings

- **Expansion Over Partitioning**: A larger area, extra bathrooms, and floors
  (stories) add massive value. In contrast, simply adding more bedrooms within
  the same area has a small impact on price (+3.05 Lakhs).
- **Premium Amenities**: Air conditioning and hot water heating are highly
  valued, indicating that buyers in this market place a high premium on climate
  control and baseline utilities.
- **Furnishing Staging Premium**: An unfurnished property faces a major discount
  (-4.28 Lakhs) compared to a furnished one, which is significantly larger than
  the cost of typical furnishing, highlighting the business value of staging.

### 9.2 Real Estate Business Recommendations

1. **For Homeowners looking to Sell**: Stage the home with attractive furnishing
   and install an air conditioning unit before listing. This relatively low
   capital expenditure can increase list prices by 4.28 to 11.25 Lakhs.
2. **For Property Developers**: Focus on building multi-story properties with
   larger floor plans and multiple bathrooms (e.g. 3-bedroom, 2-bathroom,
   2-story configurations) rather than maximizing the bedroom count on a small
   footprint.
3. **For Fix-and-Flip Investors**: Target large, unfurnished houses with low
   bathroom-to-bedroom ratios. Renovate to add bathrooms, install air
   conditioning, and stage the property to capture a high valuation markup.

---

## 10. Conclusion & Future Improvements

The Multiple Linear Regression model built on Delhi housing data provides a
robust valuation tool, explaining 74.52% of price variance. Preprocessing steps
(IQR outlier treatment and min-max scaling) were key to achieving a stable OLS
fit.

### Future Scope for Improvements

1. **Multicollinearity Diagnostics (VIF)**: Incorporate a rolling variance
   inflation factor (VIF) check to drop features with VIF > 5. Features like
   bedrooms and area might exhibit multicollinearity.
2. **Regularization (Ridge/Lasso)**: Apply L1 (Lasso) or L2 (Ridge) regression
   to penalize large coefficients, automatically perform feature selection, and
   prevent overfitting.
3. **Non-Linear Modeling**: Implement tree-based ensemble methods (e.g., Random
   Forest, Gradient Boosting, or XGBoost) to capture non-linear feature
   interactions (such as the joint impact of area and preferred location), which
   would likely push prediction accuracy past 85%.
4. **Geographical/Spatial Data**: Incorporate GPS coordinates or
   sub-neighborhood categories, as location is the ultimate driver in real
   estate pricing.
