# Loan Approval Fairness Analysis

A machine learning pipeline for loan approval prediction that evaluates both predictive performance and gender fairness, using a synthetic dataset of 45,000 loan applications.

**Course project, CS465: Machine Learning**

## Overview

Automated loan approval systems can inherit or amplify societal biases present in training data, with real socioeconomic consequences for access to credit. This project trains and compares four classifiers on a synthetic loan dataset, then evaluates not just accuracy but whether the models produce disparate outcomes across gender, a protected attribute deliberately excluded from model training and used only for post-hoc fairness evaluation.

## Results

**Tuned model performance:**

| Model | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** (best) | 0.9366 | 0.8477 | 0.9787 |
| Random Forest | 0.9300 | 0.8309 | 0.9756 |
| Decision Tree | 0.9198 | 0.8038 | 0.9573 |
| Logistic Regression | 0.9031 | 0.7781 | 0.9575 |

**Fairness results (gender):**

| Model | Demographic Parity Diff. | Disparate Impact Ratio | Equal Opportunity Diff. |
|---|---|---|---|
| Logistic Regression | 0.000150 | 1.000699 | 0.002103 |
| Decision Tree | 0.007194 | 0.968515 | 0.000031 |
| Random Forest | 0.002511 | 0.986983 | 0.006123 |
| Gradient Boosting | 0.005812 | 0.970142 | 0.008965 |

All models show close-to-zero demographic parity and equal opportunity differences, and a disparate impact ratio near 1.0 (the ideal value), indicating minimal gender-based disparity in this synthetic dataset. Logistic Regression showed the most balanced fairness profile, while Gradient Boosting and Random Forest traded a marginally higher (but still low) disparity for stronger predictive performance.

## Key Findings

1. **Gradient Boosting is the best overall model** after hyperparameter tuning, with Random Forest a close, highly competitive alternative.
2. **Ensemble methods consistently outperform Logistic Regression and Decision Tree**, both at baseline and after tuning, suggesting the data contains relationships that benefit from more flexible models.
3. **Class imbalance (77.78% rejected / 22.22% approved) is a central challenge.** All models perform better on the majority class; minority-class recall was the hardest metric to improve, which is why F1-score and ROC-AUC were emphasized over raw accuracy.
4. **Predictive performance and fairness were not in conflict here.** The strongest models (Gradient Boosting, Random Forest) achieved high accuracy without meaningfully worse fairness metrics than the simpler baselines.
5. **Results reflect a synthetic dataset with near-equal raw approval rates by gender** (22.25% for women, 22.20% for men), so these findings describe model behavior under controlled conditions, not a claim about real-world lending discrimination.

## Approach

**Dataset:** 45,000 synthetic loan applications (20,159 women, 24,841 men), 13 input features plus a binary loan status target (77.78% rejected, 22.22% approved). No missing values; a small number of age outliers (7 records over 100) were retained to preserve realistic variation.

**Preprocessing:** unified scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage. Numerical features: mean imputation + `StandardScaler`. Categorical features: mode imputation + one-hot encoding (`drop='first'`, `handle_unknown='ignore'`). Stratified 80/20 train-test split.

**Feature engineering** (13 → 17 features):
- `log_income`: log-transformed income to reduce right skew
- `high_debt_ratio`: flags loan-to-income ratio above 30%
- `employment_stability`: flags 5+ years of employment experience
- `credit_score_category`: ordinal FICO bands (poor/fair/good/excellent)

**Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, each evaluated at baseline and after `GridSearchCV` hyperparameter tuning with 5-fold stratified cross-validation.

**Fairness metrics:** Demographic Parity Difference (DPD), Disparate Impact Ratio (DIR), and Equal Opportunity Difference (EOD), computed across gender groups on model predictions.

## Tech Stack

Python, scikit-learn, pandas, matplotlib/seaborn

## Project Structure

- `notebook/` — loan_approval_fairness_analysis.ipynb
- `report/` — loan-approval-fairness-analysis-report.pdf
- `presentation/` — loan-approval-fairness-analysis-slides.pdf
- `data/` — loan_data.csv
- `README.md`

## Team

Built as a group project with Layan Almuqhem for CS465 (Machine Learning), Prince Sultan University.

## References

Full citation list available in the project report.
