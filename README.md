# Credit Card Spend Prediction — ML Competition

🏆 **Ranked #7 out of 354 participants (Top 2%)** on the private leaderboard — RMSE: 247.33

A machine learning competition project predicting customers' **monthly credit card spend** from demographic, financial, and behavioral transaction data.

https://www.kaggle.com/competitions/n463372 (see private leaderboard)

---

## Problem

Given a dataset of credit card customers (demographics, account details, transaction behavior), predict each customer's **monthly spend** as accurately as possible, evaluated by RMSE on a held-out scoring set.

## Data

- Customer-level dataset combining **numeric** (income, credit limit, transaction counts, reward points), **binary**, and **categorical** features (e.g., card type, occupation)
- Key numeric variables were strongly **right-skewed**, indicating high-value outliers and a long-tail customer base
- No severe multicollinearity among top predictors (income, credit limit, avg. transaction value, reward points balance) — each contributed distinct signal

## Approach

I explored two parallel modeling paths (documented in two notebooks):

**Notebook A — Final Submission**
- Feature engineering to create behavioral/financial ratio features
- One-hot encoding with consistent train/scoring feature alignment
- Feature selection: correlation filtering + forward Sequential Feature Selection (SFS)
- Feature extraction: variance thresholding + standardization
- **Final model: Linear Regression**, selected after comparing it against Ridge, Lasso, Decision Trees, Random Forest, Bagging, AdaBoost, and Gradient Boosting

**Notebook B — Exploratory Path**
- Alternative preprocessing using **PCA** for dimensionality reduction
- Broader model sweep: Linear/Lasso/Ridge, Decision Trees (simple + pruned via grid search), Voting Regressor, Random Forest, Bagging, AdaBoost, Gradient Boosting
- Used to benchmark whether nonlinear/ensemble methods or PCA could beat the linear approach (they didn't)

## Key Results & Insights

- **Linear Regression outperformed every nonlinear and ensemble model tested** — spending behavior in this dataset was predominantly linear, so tree-based models tended to overfit (very low train RMSE, much higher test RMSE) or underfit when kept simple
- **PCA added complexity without improving RMSE** and reduced interpretability, so it was dropped from the final pipeline
- Careful **feature engineering and selection mattered more than model complexity** for this problem

### What I'd do differently
- Automate feature engineering to capture richer nonlinear interactions
- Skip PCA unless it's clearly needed, to preserve interpretability
- Ensemble multiple strong models for a marginally better RMSE
- Handle outliers more deliberately for more stable predictions

*(Full reflection in the [presentation slides](./PAC_Presentation_Slides.pdf).)*

## Repo Structure

```
credit-card-spend-prediction/
├── README.md
├── PAC_Presentation_Slides.pdf         # Results & reflections deck
└── notebooks/
    ├── final_model_notebook_A.ipynb    # Final submitted model (Linear Regression)
    └── exploratory_model_notebook_B.ipynb  # Exploratory path (PCA, trees, ensembles)
```

## Tools

`Python` · `pandas` · `scikit-learn` (Linear/Ridge/Lasso Regression, Decision Trees, Random Forest, Bagging, AdaBoost, Gradient Boosting) · `PCA` · `matplotlib`/`seaborn`

## Project Context

Built as part of a graduate-level machine learning competition at **Columbia University**, competing against 354 participants.
