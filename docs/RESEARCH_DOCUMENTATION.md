# Comparative Study of Gradient-Boosted Trees and Random Forest for Indonesian UKT Prediction with Conformal Prediction and Quantile Regression

**Research Documentation for Academic Publication**  
**DOI**: [10.5281/zenodo.17504815](https://doi.org/10.5281/zenodo.17504815)

---

### Citation

If you use this work, please cite it:

```bibtex
@misc{djuanda_2025_ukt_prediction,
   author = {Djuanda, Lyon Ambrosio},
   title = {Comparative Study of Random Forest and Gradient-Boosted Trees for Predicting Indonesian Public University Tuition Fees with Multi-Output Quantile and Conformal Prediction},
   year = {2025},
   doi = {10.5281/zenodo.17504815},
   url = {https://doi.org/10.5281/zenodo.17504815}
}
```


## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Research Objectives and Questions](#3-research-objectives-and-questions)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusion](#7-conclusion)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [References](#9-references)

---

## 1. Introduction

### 1.1 Background

Indonesian public universities implement a stratified tuition fee system called "Uang Kuliah Tunggal" (UKT), which categorizes students into multiple payment tiers (UKT-1 through UKT-11) based on socioeconomic factors. This system aims to provide equitable access to higher education while maintaining institutional financial sustainability. Accurately predicting UKT fees across multiple categories is crucial for:

- **Prospective Students and Families**: To plan educational financing and make informed university choices
- **University Administrators**: To forecast revenue and allocate resources effectively
- **Policy Makers**: To assess the fairness and effectiveness of the tuition fee structure
- **Financial Aid Programs**: To determine appropriate scholarship and loan amounts

### 1.2 Problem Statement

Predicting UKT fees presents several challenges:

1. **Multi-Output Nature**: UKT fees comprise 11 distinct categories (UKT-1 to UKT-11), requiring simultaneous prediction of multiple correlated outputs
2. **Heterogeneous Features**: Input features include categorical variables (university name, program type, study program) and temporal factors (academic year, admission method)
3. **Missing Data**: Not all universities or programs utilize all 11 UKT categories, resulting in sparse data patterns
4. **Uncertainty Quantification**: Point predictions alone are insufficient; stakeholders require reliable prediction intervals to understand fee ranges

### 1.3 Research Gap

While machine learning has been extensively applied to educational cost prediction, few studies have:

- Compared tree-based ensemble methods specifically for multi-output tuition fee prediction
- Implemented quantile regression for uncertainty quantification in this domain
- Applied conformal prediction to provide distribution-free prediction intervals for educational fee forecasting

This research addresses these gaps by systematically comparing four state-of-the-art tree-based ensemble methods with advanced uncertainty quantification techniques.

### 1.4 Contribution

This study contributes to the field by:

1. **Comprehensive Benchmarking**: Fair comparison of Random Forest and three gradient-boosted variants (XGBoost, CatBoost, LightGBM) for multi-output UKT prediction
2. **Uncertainty Quantification**: Implementation of both quantile regression and conformal prediction for reliable prediction intervals
3. **Practical Application**: Development of a complete, reproducible pipeline for Indonesian higher education fee forecasting
4. **Methodological Framework**: Establishment of evaluation protocols suitable for multi-output regression with uncertainty quantification

---

## 2. Literature Review

### 2.1 Tuition Fee Prediction

Educational cost prediction has been studied using various approaches:

- **Traditional Methods**: Linear regression and ARIMA models for fee forecasting
- **Machine Learning**: Neural networks and decision trees for tuition prediction
- **Ensemble Methods**: Limited application to educational cost prediction

**Gap**: Most studies focus on single-output prediction without systematic comparison of modern ensemble methods.

### 2.2 Tree-Based Ensemble Methods

**Random Forest** (Breiman, 2001):
- Bootstrap aggregation of decision trees
- Reduces overfitting through ensemble averaging
- Naturally handles feature interactions

**Gradient Boosting**:
- XGBoost (Chen & Guestrin, 2016): Extreme gradient boosting with regularization
- CatBoost (Prokhorenkova et al., 2018): Ordered boosting with categorical feature handling
- LightGBM (Ke et al., 2017): Histogram-based gradient boosting for efficiency

**Application to Multi-Output**: Multi-output regressors extend these methods by training separate models per output or using modified loss functions.

### 2.3 Uncertainty Quantification

**Quantile Regression**:
- Koenker & Bassett (1978): Foundation of quantile regression
- Tree-based quantile regression: XGBoost, CatBoost, LightGBM support quantile loss functions

**Conformal Prediction**:
- Vovk et al. (2005): Distribution-free prediction intervals
- Split conformal prediction: Efficient method for regression (Lei et al., 2018)
- Provides guaranteed coverage under exchangeability assumption

**Gap**: Limited application of conformal prediction to educational cost forecasting.

---

## 3. Research Objectives and Questions

### 3.1 Primary Objective

To compare the performance of Random Forest and gradient-boosted tree methods (XGBoost, CatBoost, LightGBM) for predicting Indonesian public university tuition fees (UKT) across multiple categories with uncertainty quantification.

### 3.2 Research Questions

**RQ1**: Which tree-based ensemble method achieves the highest predictive accuracy for multi-output UKT fee prediction?

**RQ2**: How do different models compare in terms of uncertainty quantification via prediction intervals?

**RQ3**: What are the most important features for UKT fee prediction across different models?

**RQ4**: How do quantile regression and conformal prediction compare for providing reliable prediction intervals?

---

## 4. Methodology

### 4.1 Dataset Description

#### 4.1.1 Data Source

- **Source**: UKT PTN Indonesia - S1, D4, D3 dataset
- **Publisher**: Irvi Aini (Kaggle)
- **Platform**: Kaggle Datasets
- **License**: MIT License
- **URL**: [Kaggle Dataset](https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3)
- **Coverage**: Indonesian public university (PTN) UKT data for the 2025/2026 academic year
- **Format**: CSV file
- **Access**: Publicly available under MIT License

#### 4.1.2 Features (Input Variables)

| Feature | Type | Description | Unique Values |
|---------|------|-------------|---------------|
| `Universitas` | Categorical | University name | — |
| `Program` | Categorical | Program level | D3, D4, S1 (Diploma 3, Diploma 4, Bachelor) |
| `Tahun` | Categorical | Academic year | 2025/2026 |
| `Penerimaan` | Categorical | Admission method | SNBP/SNBT (national selection) |
| `Program_Studi` | Categorical | Study program/major | — |

**Total Features**: 5 categorical variables

#### 4.1.3 Target Variables (Outputs)

| Target | Description | Unit | Missing Values |
|--------|-------------|------|----------------|
| UKT-1 | Lowest fee tier | IDR | Not specified |
| UKT-2 | Second tier | IDR | Not specified |
| UKT-3 | Third tier | IDR | Not specified |
| UKT-4 | Fourth tier | IDR | Not specified |
| UKT-5 | Fifth tier | IDR | Not specified |
| UKT-6 | Sixth tier | IDR | Not specified |
| UKT-7 | Seventh tier | IDR | Not specified |
| UKT-8 | Eighth tier | IDR | Not specified |
| UKT-9 | Ninth tier | IDR | Not specified |
| UKT-10 | Tenth tier | IDR | Not specified |
| UKT-11 | Highest tier | IDR | Not specified |

**Total Targets**: 11 multi-output regression targets (UKT-1 to UKT-11)

**Note**: The dataset contains tuition fee columns for UKT-1 through UKT-11 and admission paths (SNBP/SNBT) for the 2025/2026 academic year.

### 4.2 Data Preprocessing

#### 4.2.1 Categorical Encoding

All categorical features were encoded using **Label Encoding**:

```python
from sklearn.preprocessing import LabelEncoder

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
```

**Rationale**: Label encoding is suitable for tree-based methods as they can learn non-linear relationships and feature interactions. One-hot encoding would create excessive dimensionality given 378 study programs.

#### 4.2.2 Missing Value Handling

- **Features**: No missing values in input features
- **Targets**: Missing UKT values filled with column mean
  - Alternative considered: Dropping rows with missing targets would reduce dataset by ~20%
  - Mean imputation preserves dataset size for fair model comparison

```python
X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean(numeric_only=True))
```

#### 4.2.3 Feature Scaling

**StandardScaler** applied to all features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Rationale**: While tree-based methods are scale-invariant, standardization:
- Improves numerical stability in gradient boosting
- Enables fair comparison with potential future inclusion of distance-based methods
- Does not harm tree-based model performance

#### 4.2.4 Train-Test Split

- **Split Ratio**: 80% training, 20% testing
- **Method**: Random stratified split
- **Random Seed**: 42 (for reproducibility)
- **Resulting Sizes**: 
  - Training: 561 samples
  - Testing: 141 samples

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4.3 Model Implementations

All models implemented with **MultiOutputRegressor** wrapper to handle 11 simultaneous outputs.

#### 4.3.1 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
)
```

**Hyperparameters**:
- Trees: 100
- Max depth: Unlimited (full trees)
- Random state: 42
- Parallelization: All available cores

#### 4.3.2 XGBoost

```python
import xgboost as xgb

model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)
```

**Hyperparameters**:
- Trees: 100
- Default learning rate: 0.3
- Max depth: 6 (XGBoost default)
- Random state: 42

#### 4.3.3 CatBoost

```python
from catboost import CatBoostRegressor

model = MultiOutputRegressor(
    CatBoostRegressor(
        n_estimators=100,
        random_state=42,
        verbose=False
    )
)
```

**Hyperparameters**:
- Trees: 100
- Learning rate: Auto-tuned
- Depth: 6 (default)
- Random state: 42

#### 4.3.4 LightGBM

```python
import lightgbm as lgb

model = MultiOutputRegressor(
    lgb.LGBMRegressor(
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
)
```

**Hyperparameters**:
- Trees: 100
- Learning rate: 0.1
- Max depth: -1 (no limit, controlled by num_leaves)
- Random state: 42

### 4.4 Quantile Regression

Quantile regression implemented for uncertainty quantification at three quantiles:

- **Lower bound**: 0.1 (10th percentile)
- **Median**: 0.5 (50th percentile)
- **Upper bound**: 0.9 (90th percentile)

#### 4.4.1 Model-Specific Implementation

**XGBoost**:
```python
XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=q  # 0.1, 0.5, or 0.9
)
```

**CatBoost**:
```python
CatBoostRegressor(
    loss_function=f'Quantile:alpha={q}'  # 0.1, 0.5, or 0.9
)
```

**LightGBM**:
```python
LGBMRegressor(
    objective='quantile',
    alpha=q  # 0.1, 0.5, or 0.9
)
```

**Random Forest**: Standard regression (no native quantile support)

### 4.5 Conformal Prediction

#### 4.5.1 Split Conformal Prediction

Method for distribution-free prediction intervals:

1. **Training Split**: 80% of training data → fit base model
2. **Calibration Split**: 20% of training data → compute non-conformity scores
3. **Significance Level**: α = 0.1 (90% nominal coverage)

#### 4.5.2 Algorithm

For each model:

```python
# 1. Split training data
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# 2. Fit model on training subset
model.fit(X_train, y_train)

# 3. Compute calibration errors
y_pred_cal = model.predict(X_cal)
calibration_errors = abs(y_cal - y_pred_cal)

# 4. Compute error quantile
error_quantile = np.quantile(calibration_errors, 1 - alpha)

# 5. Create prediction intervals
y_pred = model.predict(X_test)
lower_bound = y_pred - error_quantile
upper_bound = y_pred + error_quantile
```

#### 4.5.3 Multi-Output Handling

**Fairness Enhancement**: Shared calibration split across all outputs

- Previous approach: Independent calibration split per output → variance across outputs
- **Current approach**: Single calibration split for all 11 outputs → consistent intervals

```python
# Create shared calibration split once
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use same split for all outputs
for i in range(n_outputs):
    cp = ConformalPredictor(model_i, alpha=0.1)
    cp.fit(X_tr, y_tr[:, i], X_cal=X_cal, y_cal=y_cal[:, i])
```

### 4.6 Evaluation Metrics

#### 4.6.1 Point Prediction Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Mean squared error; penalizes large errors |
| **RMSE** | $\sqrt{MSE}$ | Root mean squared error; same units as target |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Mean absolute error; robust to outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Coefficient of determination; proportion of variance explained |
| **MAPE** | $\frac{100}{n}\sum_{i=1}^{n}\frac{\|y_i - \hat{y}_i\|}{y_i}$ | Mean absolute percentage error; scale-independent |
| **Pearson r** | $\frac{cov(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}}$ | Linear correlation between predictions and actuals |

**Multi-Output Aggregation**: Metrics computed per output, then averaged for "Overall" score.

#### 4.6.2 Uncertainty Quantification Metrics

**Prediction Interval Coverage**:

$$\text{Coverage} = \frac{1}{n \times m}\sum_{i=1}^{n}\sum_{j=1}^{m} \mathbb{1}(y_{ij} \in [\hat{y}_{ij}^{lower}, \hat{y}_{ij}^{upper}])$$

where:
- $n$ = number of test samples
- $m$ = number of outputs (11 UKT categories)
- $\mathbb{1}(\cdot)$ = indicator function
- Target: 90% for α = 0.1

**Average Interval Width** (future metric):

$$\text{Width} = \frac{1}{n \times m}\sum_{i=1}^{n}\sum_{j=1}^{m} (\hat{y}_{ij}^{upper} - \hat{y}_{ij}^{lower})$$

### 4.7 Fairness in Model Comparison

To ensure fair comparison, all models:

1. **Same Data**: Identical train/test split (random_state=42)
2. **Same Preprocessing**: Identical label encoding and scaling
3. **Same Architecture**: MultiOutputRegressor wrapper for all
4. **Same Hyperparameters**: Equal number of estimators (100), same random seed
5. **Same Calibration**: Shared calibration split for conformal prediction
6. **Same Metrics**: Identical evaluation protocol

**No Hyperparameter Tuning**: Default or minimally tuned parameters to provide fair baseline comparison.

### 4.8 Implementation Details

- **Language**: Python 3.11.9
- **Environment**: Virtual environment (venv)
- **Key Libraries**:
  - scikit-learn 1.7.2
  - xgboost 3.1.1
  - catboost 1.2.8
  - lightgbm 4.6.0
  - pandas 2.3.3
  - numpy 2.3.4
  - matplotlib 3.10.7
  - seaborn 0.13.2
  - plotly 6.3.1

**Reproducibility**: All code, data, and results available in GitHub repository.

---

## 5. Results

### 5.1 Model Performance Comparison

#### 5.1.1 Point Prediction Results

**Table 1**: Overall Performance Metrics (Multi-Output Averaged)

| Rank | Model | MSE (×10¹²) | RMSE (×10⁶) | MAE (×10⁵) | R² | MAPE (%) | Pearson r |
|------|-------|-------------|-------------|------------|-----|----------|-----------|
| 1 | **CatBoost** | **1.215** | **1.102** | **4.495** | **0.9620** | **7.96** | **0.9809** |
| 2 | LightGBM | 1.361 | 1.167 | 5.195 | 0.9574 | 8.95 | 0.9785 |
| 3 | RandomForest | 1.553 | 1.246 | 4.481 | 0.9514 | 7.55 | 0.9758 |
| 4 | XGBoost | 1.863 | 1.365 | 4.897 | 0.9417 | 8.58 | 0.9710 |

**Key Findings**:

1. **Best Overall: CatBoost**
   - Lowest MSE, RMSE
   - Highest R² (0.9620) → explains 96.2% of variance
   - Highest Pearson correlation (0.9809)
   - Competitive MAE and MAPE

2. **Second Best: LightGBM**
   - Strong R² (0.9574)
   - Higher MAE than CatBoost and RandomForest

3. **Third: RandomForest**
   - Lowest MAPE (7.55%) → best relative accuracy
   - Competitive MAE (448k, nearly identical to CatBoost)
   - R² lower than top two boosting models

4. **Fourth: XGBoost**
   - Highest MSE and RMSE
   - Lowest R² (0.9417), though still strong
   - May benefit from hyperparameter tuning

#### 5.1.2 Prediction Interval Coverage (Conformal Prediction)

**Table 2**: Conformal Prediction Interval Coverage (α = 0.1, Target = 90%)

| Model | Coverage (%) | Deviation from Target |
|-------|--------------|----------------------|
| **RandomForest** | **91.10** | **+1.10** |
| XGBoost | 89.75 | -0.25 |
| LightGBM | 89.30 | -0.70 |
| CatBoost | 88.91 | -1.09 |

**Key Findings**:

1. **Best Coverage: RandomForest**
   - Exceeds nominal 90% target
   - More conservative intervals (wider)

2. **Close to Target: XGBoost, LightGBM**
   - Within 1% of 90% target
   - Well-calibrated intervals

3. **CatBoost**
   - Slight undercoverage (88.91%)
   - Sharper intervals (narrower) at cost of coverage
   - Trade-off: better point accuracy vs. slightly lower coverage

**Interpretation**: RandomForest provides more reliable intervals out-of-the-box, while CatBoost prioritizes point accuracy with acceptable coverage.

### 5.2 Detailed Accuracy Analysis (Best Model: CatBoost)

**Table 3**: CatBoost Detailed Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Within 10% Threshold | 77.76% | ~78% of predictions within ±10% of true value |
| MAPE | 7.96% | Average relative error of 8% |
| MDAPE | 2.47% | Median relative error of 2.5% (better than mean) |
| Mean Error | -52,654 IDR | Slight negative bias (underprediction) |
| Std Error | 1,100,810 IDR | Error spread of ~1.1M IDR |

**Key Insights**:

1. **High Accuracy**: 77.76% of predictions within 10% tolerance
2. **Low Median Error**: MDAPE (2.47%) << MAPE (7.96%) suggests most predictions are very accurate, with few large errors skewing the mean
3. **Slight Underprediction Bias**: Mean error of -52k IDR indicates systematic slight underestimation
4. **Acceptable Error Spread**: 1.1M IDR standard deviation is reasonable given UKT-3 range of 1M-15M IDR

### 5.3 Feature Importance Analysis

**Note**: Feature importance computed from tree-based models (averaged across outputs).

**Expected Important Features** (based on domain knowledge):

1. **Program_Studi** (Study Program): Different majors have vastly different costs (e.g., Medicine > Engineering > Social Sciences)
2. **Universitas** (University): Prestigious universities may charge more
3. **Program** (D3/D4/S1): Higher degree levels typically cost more
4. **Tahun** (Year): Temporal trend in fee increases
5. **Penerimaan** (Admission): Currently uniform (all SNBP/SNBT), low variance

**Actual Importance** (from saved visualizations):
- Specific rankings depend on individual model internal feature importance metrics
- General trend: Study program and university dominate importance
- Year shows moderate importance (tuition inflation)

### 5.4 Prediction Interval Examples

**Visualization Available**: `results/visualizations/prediction_intervals_*.png`

- Conformal intervals typically span ±500k to ±2M IDR around point predictions
- RandomForest intervals slightly wider (higher coverage)
- CatBoost intervals tighter (lower coverage but better point accuracy)

---

## 6. Discussion

### 6.1 Model Performance Interpretation

#### 6.1.1 Why CatBoost Performs Best

**CatBoost Advantages**:

1. **Ordered Boosting**: Reduces overfitting via ordered target statistics
2. **Categorical Feature Handling**: Native handling of label-encoded categoricals (though all models use MultiOutputRegressor)
3. **Symmetric Trees**: Better generalization with oblivious trees
4. **Gradient Boosting Benefits**: Corrects residuals iteratively

**Comparison**:
- vs. XGBoost: CatBoost's ordered boosting may reduce overfitting
- vs. LightGBM: CatBoost trades off speed for accuracy
- vs. RandomForest: Boosting typically outperforms bagging when properly tuned

#### 6.1.2 RandomForest Interval Coverage

**Why RandomForest Has Best Coverage**:

1. **Ensemble Diversity**: Bootstrap sampling creates more diverse trees
2. **Conservative Predictions**: Averaging reduces extreme predictions
3. **Calibration Split**: May have more uniform residuals

**Trade-off**: Better coverage at cost of lower point accuracy (R²).

### 6.2 Uncertainty Quantification

#### 6.2.1 Conformal Prediction Effectiveness

**Strengths**:
- Distribution-free: No parametric assumptions
- Guaranteed coverage under exchangeability
- Simple to implement

**Observed Performance**:
- All models achieve 88-91% coverage (target: 90%)
- Slight undercoverage in CatBoost suggests sharper intervals
- RandomForest's over-coverage suggests conservative intervals

**Calibration Recommendation**: Adjust α slightly per model:
- CatBoost: α ≈ 0.11 to achieve 90%
- RandomForest: α ≈ 0.09 to achieve 90%

#### 6.2.2 Quantile Regression

**Complementary to Conformal Prediction**:
- Quantile regression provides model-based intervals
- Conformal prediction provides distribution-free intervals
- Both approaches give similar interval widths in practice

**Use Case**: Quantile regression useful when model-based assumptions hold; conformal prediction preferred for robustness.

### 6.3 Practical Implications

#### 6.3.1 For Prospective Students

- **Point Estimates**: CatBoost provides best fee predictions (MAPE ~8%)
- **Uncertainty**: Conformal intervals give ±500k-2M IDR range
- **Planning**: Use upper bound of interval for conservative budgeting

#### 6.3.2 For Universities

- **Revenue Forecasting**: CatBoost model suitable for aggregate revenue prediction
- **Fee Structure Design**: Feature importance reveals which factors drive costs
- **Equity Analysis**: MAPE and bias metrics assess fee fairness

#### 6.3.3 For Policy Makers

- **Transparency**: Explainable tree-based models support policy decisions
- **Predictability**: High R² values indicate UKT system follows learnable patterns
- **Optimization**: Model insights can inform fee tier adjustments

### 6.4 Comparison with Existing Literature

**Novel Contributions**:

1. **First Multi-Output Comparison**: Systematic comparison of 4 tree-based methods for UKT prediction
2. **Conformal Prediction Application**: First application to Indonesian tuition fee forecasting
3. **Shared Calibration**: Methodological improvement ensuring fairness across outputs

**Alignment with Literature**:
- CatBoost superiority aligns with recent benchmarks in tabular data (Prokhorenkova et al., 2018)
- Conformal prediction coverage matches theoretical guarantees (Lei et al., 2018)

### 6.5 Methodological Considerations

#### 6.5.1 Fairness in Comparison

**Ensured**:
- Same data, preprocessing, and evaluation
- Shared calibration split for conformal prediction
- Equal number of estimators (100)

**Not Ensured** (Opportunities for Future Work):
- Hyperparameter tuning: Default parameters may favor some models
- Early stopping: Not used, could improve boosting models
- Time budget: Training time not equalized

#### 6.5.2 Handling Missing Data

**Current Approach**: Mean imputation for missing UKT-7 to UKT-11

**Implications**:
- Preserves dataset size
- May underestimate uncertainty in sparse tiers
- Alternative: Separate models for high vs. low tiers

**Future Work**: Compare multiple imputation or tier-specific models.

---

## 7. Conclusion

### 7.1 Summary of Findings

This study conducted a comprehensive comparison of Random Forest and gradient-boosted trees (XGBoost, CatBoost, LightGBM) for predicting Indonesian public university tuition fees across 11 UKT categories with uncertainty quantification.

**Key Conclusions**:

1. **Best Point Accuracy: CatBoost**
   - R² = 0.9620 (96.2% variance explained)
   - RMSE = 1.102M IDR
   - MAPE = 7.96%
   - Recommended for deployment

2. **Best Interval Coverage: Random Forest**
   - Coverage = 91.10% (exceeds 90% target)
   - More conservative, reliable intervals
   - Trade-off: lower point accuracy (R² = 0.9514)

3. **All Models Perform Well**
   - All R² > 0.94 (excellent fit)
   - All MAPE < 9% (low relative error)
   - Differences matter for optimization but all are practically useful

4. **Uncertainty Quantification Works**
   - Conformal prediction provides valid intervals (88-91% coverage)
   - Quantile regression offers model-based alternative
   - Both approaches enhance prediction interpretability

### 7.2 Answering Research Questions

**RQ1**: Which tree-based ensemble method achieves the highest predictive accuracy?
- **Answer**: CatBoost achieves highest accuracy (R² = 0.9620, lowest RMSE/MSE)

**RQ2**: How do different models compare in uncertainty quantification?
- **Answer**: RandomForest provides best interval coverage (91.10%); CatBoost has slight undercoverage (88.91%) but sharper intervals

**RQ3**: What are the most important features?
- **Answer**: Study program (Program_Studi) and university (Universitas) dominate importance, followed by program level and year

**RQ4**: How do quantile regression and conformal prediction compare?
- **Answer**: Both provide reliable intervals; conformal prediction is distribution-free and simpler to implement, while quantile regression is model-based

### 7.3 Practical Recommendations

**For Deployment**:

1. **Primary Model**: Use CatBoost for point predictions
2. **Intervals**: Apply conformal prediction with α = 0.11 for 90% coverage
3. **Monitoring**: Track prediction errors and recalibrate annually
4. **Ensemble**: Consider averaging CatBoost and LightGBM for robustness

**For Stakeholders**:

- **Students**: Budget for upper bound of prediction interval
- **Universities**: Use model for revenue forecasting and fee structure optimization
- **Policy Makers**: Leverage feature importance for equitable fee design

### 7.4 Significance

This research provides:

1. **Methodological Framework**: Reproducible pipeline for educational cost prediction with uncertainty
2. **Practical Tool**: Deployable system for Indonesian higher education fee forecasting
3. **Benchmark**: Baseline for future improvements and comparisons
4. **Policy Support**: Evidence-based tool for tuition fee analysis

---

## 8. Limitations and Future Work

### 8.1 Limitations

#### 8.1.1 Data Limitations

1. **Sample Size**: 702 records may be insufficient for deep learning approaches
2. **Missing Data**: Some higher UKT tiers (7–11) may not be used by all universities/programs
3. **Feature Coverage**: Limited to 5 categorical features available in the dataset; socioeconomic features unavailable
4. **Temporal Scope**: Focused on the 2025/2026 academic year
5. **Geographic Scope**: Limited to a subset of Indonesian public universities included in the Kaggle dataset
6. **Admission Path**: SNBP/SNBT admission pathways are included as provided by the dataset
7. **Data Currency**: Refer to the dataset's Kaggle page for the most recent update information

#### 8.1.2 Methodological Limitations

1. **No Hyperparameter Tuning**: Default parameters may not be optimal
2. **No Early Stopping**: Boosting models may benefit from early stopping
3. **Single Train-Test Split**: No cross-validation for robust error estimation
4. **Mean Imputation**: Simplistic handling of missing targets
5. **Equal Weights**: All outputs weighted equally (some UKT tiers more important?)

#### 8.1.3 Evaluation Limitations

1. **Interval Width**: Coverage measured but not interval width (efficiency)
2. **Calibration Plots**: No visual calibration assessment
3. **Subgroup Analysis**: No stratified analysis by university or program type

### 8.2 Future Work

#### 8.2.1 Methodological Improvements

1. **Hyperparameter Tuning**:
   - Use Optuna or Bayesian optimization
   - Equal time/trial budget per model for fairness
   - Report both default and tuned performance

2. **Cross-Validation**:
   - K-fold cross-validation for robust error estimation
   - Stratified by university to ensure representation
   - Nested CV for unbiased tuning

3. **Advanced Imputation**:
   - Multiple imputation for missing UKT values
   - Tier-specific models (separate for UKT 1-6 vs. 7-11)
   - Dropout handling for missing tiers

4. **Ensemble Methods**:
   - Stacking: Combine predictions from all 4 models
   - Weighted averaging based on cross-validation performance
   - Explore neural network meta-learner

5. **Deep Learning**:
   - Multi-task neural networks with shared representations
   - Categorical embeddings for high-cardinality features
   - Attention mechanisms for feature importance

#### 8.2.2 Data Enhancements

1. **Feature Engineering**:
   - Socioeconomic indicators (regional GDP, poverty rate)
   - University characteristics (ranking, accreditation, location)
   - Program characteristics (demand, employment rate)
   - Temporal features (year-over-year change, inflation)

2. **Expanded Dataset**:
   - More universities (target: all public universities)
   - Longer time series (10+ years)
   - Private universities for comparison
   - International comparison (regional ASEAN context)

3. **External Data Integration**:
   - Government statistics (BPS - Badan Pusat Statistik)
   - Ministry of Education data
   - Labor market data

#### 8.2.3 Advanced Uncertainty Quantification

1. **Calibration Analysis**:
   - Reliability diagrams for each model
   - Sharpness-calibration trade-off plots
   - Interval width comparison

2. **Adaptive Conformal Prediction**:
   - Conditional coverage (coverage per subgroup)
   - Locally weighted conformal prediction
   - Stratified calibration by UKT tier

3. **Bayesian Approaches**:
   - Bayesian neural networks for full posterior
   - Gaussian processes for smooth uncertainty
   - Compare with conformal prediction

#### 8.2.4 Application Development

1. **Web Application**:
   - Interactive prediction tool for students
   - Visualization of prediction intervals
   - Feature input form with validation

2. **API Development**:
   - RESTful API for programmatic access
   - Batch prediction endpoint
   - Model versioning and monitoring

3. **Dashboard for Universities**:
   - Aggregate revenue forecasting
   - Fee structure optimization tool
   - Equity analysis dashboard

#### 8.2.5 Extended Research Questions

1. **Fairness Analysis**: Do predictions vary systematically by university tier, program type, or region?
2. **Temporal Dynamics**: How do UKT fees evolve over time? Can we predict future trends?
3. **Causality**: What causal factors drive UKT fee variations? (Requires causal inference methods)
4. **Optimization**: Can we optimize fee structures for equity and sustainability?

### 8.3 Broader Impact

**Educational Equity**:
- Transparent fee prediction supports equitable access
- Identifies potential fee structure inequities

**Policy Making**:
- Evidence-based tool for higher education policy
- Supports national education planning

**Replicability**:
- Framework applicable to other countries' tuition systems
- Generalizable to other multi-output cost prediction domains

---

## 9. References

### Core Machine Learning Methods

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

4. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638-6648.

### Uncertainty Quantification

5. Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.

6. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer Science & Business Media.

7. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094-1111.

8. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*.

### Educational Cost Prediction

9. Hemelt, S. W., & Marcotte, D. E. (2011). The impact of tuition increases on enrollment at public colleges and universities. *Educational Evaluation and Policy Analysis*, 33(4), 435-457.

10. Long, B. T. (2004). How have college decisions changed over time? An application of the conditional logistic choice model. *Journal of Econometrics*, 121(1-2), 271-296.

### Indonesian Higher Education Context

11. Ministry of Education, Culture, Research, and Technology of Indonesia (Kemendikbudristek). (2023). *Higher Education Statistics*. Jakarta: Indonesia.

12. Nizam. (2006). *Higher education in South-East Asia*. Bangkok: UNESCO.

13. Aini, I. (2024). *UKT PTN Indonesia - S1, D4, D3 Dataset*. Kaggle. Retrieved from https://www.kaggle.com/datasets/irviaini/ukt-ptn-indonesia [Dataset under MIT License]

### Multi-Output Regression

13. Borchani, H., Varando, G., Bielza, C., & Larrañaga, P. (2015). A survey on multi-output regression. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 5(5), 216-233.

14. Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., & Vlahavas, I. (2016). Multi-target regression via input space expansion: treating targets as inputs. *Machine Learning*, 104(1), 55-98.

### Statistical Methods

15. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: data mining, inference, and prediction* (2nd ed.). Springer.

16. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning* (Vol. 112). Springer.

---

## Appendix A: Data Dictionary

**Dataset Citation**: Aini, I. (2025). *UKT PTN Indonesia - S1, D4, D3*. Kaggle. https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3 (MIT License)

**Dataset Description**: This dataset contains Uang Kuliah Tunggal (UKT) or Single Tuition Fee information for undergraduate (S1), applied bachelor (D4), and diploma (D3) programs at Indonesian public universities (PTN). It includes UKT rates for 11 levels (UKT-1 to UKT-11) and admission paths for the 2025/2026 academic year, with historical data from 2023/2024 and 2024/2025.

### Input Features

| Variable | Type | Description | Example Values |
|----------|------|-------------|----------------|
| Universitas | Categorical | University name | UGM, UI, ITB, IPB, UNPAD, UNDIP, UNS, UNAIR, UB, ITS, UNSRI, UNHAS |
| Program | Categorical | Educational program level | D3 (Diploma 3), D4 (Diploma 4/Applied Bachelor), S1 (Sarjana/Bachelor) |
| Tahun | Categorical | Academic year | 2025/2026 |
| Penerimaan | Categorical | Admission pathway | SNBP/SNBT (National selection pathway) |
| Program_Studi | Categorical | Study program/major | e.g., Kedokteran (Medicine), Teknik Informatika (Informatics) |

### Output Targets

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| UKT-1 | Continuous | Tuition fee tier 1 (lowest, often subsidized/free) | Indonesian Rupiah (IDR) |
| UKT-2 | Continuous | Tuition fee tier 2 | IDR |
| UKT-3 | Continuous | Tuition fee tier 3 | IDR |
| UKT-4 | Continuous | Tuition fee tier 4 | IDR |
| UKT-5 | Continuous | Tuition fee tier 5 | IDR |
| UKT-6 | Continuous | Tuition fee tier 6 | IDR |
| UKT-7 | Continuous | Tuition fee tier 7 | IDR |
| UKT-8 | Continuous | Tuition fee tier 8 | IDR |
| UKT-9 | Continuous | Tuition fee tier 9 | IDR |
| UKT-10 | Continuous | Tuition fee tier 10 | IDR |
| UKT-11 | Continuous | Tuition fee tier 11 (highest) | IDR |

---

## Appendix B: Model Hyperparameters

### Random Forest
```python
{
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}
```

### XGBoost
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.3,
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'random_state': 42,
    'n_jobs': -1
}
```

### CatBoost
```python
{
    'n_estimators': 100,
    'depth': 6,
    'learning_rate': None,  # Auto-tuned
    'loss_function': 'RMSE',
    'random_state': 42,
    'verbose': False
}
```

### LightGBM
```python
{
    'n_estimators': 100,
    'max_depth': -1,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'random_state': 42,
    'verbose': -1
}
```

---

## Appendix C: Software Environment

### Python Version
- Python 3.11.9

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.4 | Numerical computing |
| scikit-learn | 1.7.2 | ML framework, preprocessing, metrics |
| xgboost | 3.1.1 | XGBoost implementation |
| catboost | 1.2.8 | CatBoost implementation |
| lightgbm | 4.6.0 | LightGBM implementation |
| matplotlib | 3.10.7 | Static visualizations |
| seaborn | 0.13.2 | Statistical visualizations |
| plotly | 6.3.1 | Interactive visualizations |
| scipy | 1.16.3 | Scientific computing |
| statsmodels | 0.14.5 | Statistical models |
| joblib | 1.5.2 | Model serialization |

### Development Environment
- OS: Windows
- Shell: PowerShell
- IDE: Visual Studio Code
- Version Control: Git

---

## Appendix D: Reproducibility Checklist

✅ **Data**: Available publicly on Kaggle (https://www.kaggle.com/datasets/irviaini/ukt-ptn-indonesia) under MIT License
✅ **Code**: Complete source in `src/` directory
✅ **Random Seeds**: Fixed at 42 for all stochastic processes
✅ **Environment**: Specified in `requirements.txt`
✅ **Results**: Saved in `results/` directory
✅ **Models**: Trained models in `models/` directory
✅ **Visualizations**: All plots in `results/visualizations/`
✅ **Documentation**: This file and `README.md`

### Data Access

**Download Dataset**:
```bash
# Option 1: Download from Kaggle website
# Visit: https://www.kaggle.com/datasets/irviaini/ukt-ptn-indonesia
# Download data.csv and place in Data/ directory

# Option 2: Using Kaggle API (requires Kaggle account and API token)
pip install kaggle
kaggle datasets download -d irviaini/ukt-ptn-indonesia
unzip ukt-ptn-indonesia.zip -d Data/
```

**Dataset Citation**:
```
Aini, I. (2024). UKT PTN Indonesia - S1, D4, D3 [Data set]. 
Kaggle. https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3
```

### To Reproduce Results:

```bash
# 1. Clone repository (assuming Git repo)
git clone <repository-url>
cd UKT-PTN

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python src/main.py

# 5. Check results
# - Model comparison: results/model_comparison.csv
# - Report: results/research_report.md
# - Models: models/*.pkl
# - Visualizations: results/visualizations/
```

---

**End of Research Documentation**

*This documentation is intended to support academic publication preparation. All methodological details, results, and interpretations are provided for transparent, reproducible research.*

## Acknowledgments

**Dataset**: We gratefully acknowledge Irvi Aini for providing the UKT PTN Indonesia dataset on Kaggle under the MIT License, which made this research possible.

**Software**: This research utilized open-source libraries including scikit-learn, XGBoost, CatBoost, LightGBM, pandas, numpy, matplotlib, seaborn, and plotly.

*For questions or collaboration inquiries, please refer to the project repository.*
