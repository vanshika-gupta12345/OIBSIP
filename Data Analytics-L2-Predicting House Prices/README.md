# Predicting House Prices with Linear Regression

An end-to-end linear regression pipeline — from raw data through EDA, feature selection,
missing-value handling, one-hot encoding, and model evaluation, to full coefficient
interpretation and a Ridge/Lasso comparison.

> 📌 Internship project — full regression workflow from raw features to an interpretable,
> business-ready pricing model.

## 📊 Project Overview

This project answers:

- Which house features actually drive sale price, and by how much in dollar terms?
- How much of the variation in price can a linear model explain using a focused,
  interpretable feature set instead of throwing every available column at it?
- Does adding a bedroom really make a house *less* valuable — and if the data says yes,
  what's actually going on?
- Does regularisation (Ridge/Lasso) meaningfully beat plain Linear Regression here?

## 🗂️ Dataset

**Source:** [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) —
Kaggle's "House Prices: Advanced Regression Techniques" competition data, compiled by Dean
De Cock.

| | |
|---|---|
| Rows | 1,460 home sales |
| Location | Ames, Iowa, USA |
| Period | 2006–2010 |
| Columns (raw) | 81 |
| Target | `SalePrice` (mean \$180,921, median \$163,000) |

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`LinearRegression`, `RidgeCV`, `LassoCV`, `StandardScaler`, metrics)
- matplotlib, seaborn
- Jupyter Notebook

## 📁 Repository Structure

```
├── README.md                              # This file
├── House_Price_Linear_Regression.ipynb    # Full analysis notebook (executed, with outputs)
├── house_prices_dataset.csv               # Raw dataset used (all 81 original columns)
└── requirements.txt                       # Python dependencies
```

## 🔍 What's Inside the Notebook

1. **EDA** — shape, full null audit (81 columns), `SalePrice` distribution and its
   right-skew
2. **Feature selection discussion** — a deliberate 10-feature subset covering area,
   location, room count, and age (as the brief specifies), plus two additions
   (`OverallQual`, `GarageCars`) justified by their standout correlation with price
3. **Missing-value handling** — median imputation for `LotFrontage` (the only selected
   feature with real gaps, 17.7% missing), with reasoning
4. **One-hot encoding** — `Neighborhood` (25 categories) encoded with `drop_first=True`
5. **Correlation heatmap** — numeric features vs. `SalePrice`, plus a neighbourhood
   average-price bar chart (location isn't numeric, so it needs a different view)
6. **80/20 train/test split**
7. **Linear Regression** trained and evaluated: **MSE, RMSE, R²**
8. **Actual vs. predicted scatter plot** with a reference diagonal
9. **Residual plot** — including an honest discussion of the heteroscedasticity it reveals
10. **Full coefficient analysis** — including a specific, correctly-reasoned explanation of
    why `BedroomAbvGr` has a *negative* coefficient (multicollinearity, not literal causation)
11. **Bonus: Ridge and Lasso comparison** via cross-validated alpha selection, with proper
    feature standardisation (and an explanation of why scaling matters here but didn't for
    plain Linear Regression)

Every code cell is pre-executed, so every chart and metric renders immediately on
GitHub/nbviewer with no need to re-run anything.

## 💡 Key Findings

| Model | RMSE | R² |
|---|---|---|
| **Linear Regression** | **$35,892.53** | **0.8320** |
| Ridge (α=120.68, CV-tuned) | $36,441.74 | 0.8269 |
| Lasso (α=494.17, CV-tuned) | $35,941.69 | 0.8316 |

- The model explains **83.2%** of price variance using just 10 underlying features (34
  columns after encoding).
- **`OverallQual` is the single strongest lever**: +$15,902 in predicted price per point on
  its 1–10 scale.
- **Location is powerful**: the priciest neighbourhoods (`StoneBr`, `NoRidge`, `NridgHt`)
  carry a $72,000–$75,000 premium over the cheapest ones in the model's coefficients — and
  raw average prices vary more than 4x across neighbourhoods.
- **A genuinely counterintuitive result, explained rather than glossed over**: `BedroomAbvGr`
  has a *negative* coefficient (−$7,549/bedroom). This isn't "more bedrooms = less value" —
  it's a multicollinearity effect: holding total rooms and living area fixed, adding a
  bedroom usually means subdividing existing space into smaller rooms, which the model
  correctly picks up on as a partial (not standalone) relationship.
- **Regularisation doesn't clearly beat plain OLS here** — with only 34 features and 1,168
  training rows, there isn't enough overfitting for Ridge/Lasso's variance reduction to pay
  off in test RMSE. Lasso's real value-add was interpretability: it zeroed out 6 of 34
  coefficients outright.

## ✅ Conclusion

**Real-world application:** automated home valuation tools, appraisal sanity-checking
(flagging listings priced far outside what their features would predict), and renovation ROI
estimation — e.g. using the `OverallQual` coefficient to estimate the expected value uplift
from a quality-improving renovation before committing to it.

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook House_Price_Linear_Regression.ipynb
```

## 📄 License

Dataset compiled by Dean De Cock and distributed via Kaggle for educational use. Code in
this repository is shared for portfolio and educational purposes.
