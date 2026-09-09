# 🍷 Wine Quality Prediction — Classification Model Comparison

Internship project: train and compare three classification models — **Random
Forest**, **SGD (Stochastic Gradient Descent)**, and **SVC (Support Vector
Classifier)** — to predict whether a wine is "good" quality from its
physicochemical properties (acidity, density, alcohol content, etc.).

## Contents

| File | Description |
|---|---|
| `winequality-red.csv` | Raw dataset — 1,599 red wine samples, 11 physicochemical features + `quality` score |
| `wine_quality_prediction.ipynb` | Full, pre-executed analysis notebook (EDA → feature engineering → modelling → evaluation → conclusion) |
| `README.md` | This file |

## Dataset

**Wine Quality (Red)** — Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis,
J. (2009). *Modeling wine preferences by data mining from physicochemical
properties.* Decision Support Systems, 47(4). UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/186/wine+quality (also mirrored on Kaggle).
Licensed CC BY 4.0.

- **1,599 samples**, no missing values, **240 duplicate rows** (identified and
  removed during cleaning, leaving 1,359 unique samples)
- **11 input features:** fixed acidity, volatile acidity, citric acid, residual
  sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH,
  sulphates, alcohol
- **Target:** `quality`, an integer sensory rating from human tasters, ranging
  3–8 in this dataset (not the full 0–10 scale)

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook wine_quality_prediction.ipynb
```

The notebook has already been run top-to-bottom, so every table and chart is
visible without re-executing anything — but it will also re-run cleanly from
scratch if you want to reproduce or modify it.

## Methodology

1. **Load & inspect** — shape, dtypes, summary statistics, class distribution of `quality`
2. **Clean** — checked for missing values (none found) and duplicate rows (240 found and dropped)
3. **EDA** — distribution plots for all 11 features, correlation heatmap
4. **Class imbalance discussion** — quality scores 5–6 make up ~82% of the data; the extremes (3, 4, 8) are rare, which makes the raw 6-class problem hard to learn and makes accuracy a misleading metric on its own
5. **Feature engineering** — binned `quality` into:
   - a **binary** target `quality_label` (`good` if quality ≥ 7, else `bad`) — used as the primary modelling target
   - a **3-class** target `quality_class` (`low` 3–4 / `medium` 5–6 / `high` 7–8) — set up for the same pipeline, not modelled end-to-end in this version
6. **Stratified 80/20 train/test split** to preserve the ~86% bad / ~14% good ratio in both sets
7. **StandardScaler** fitted on the training set only, applied to both splits (needed for SGD and SVC, harmless for Random Forest)
8. **Trained 3 classifiers**, all with `class_weight='balanced'` to counter the imbalance: `RandomForestClassifier`, `SGDClassifier` (log-loss), `SVC` (RBF kernel)
9. **Evaluated** each with accuracy, full classification report, and confusion matrix
10. **Feature importance** chart from the Random Forest model
11. **Side-by-side comparison table** across all three models

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Recall — "good" class |
|---|---|---|---|---|---|
| **Random Forest** | **0.893** | **0.814** | 0.665 | **0.707** | 0.351 |
| SGD Classifier | 0.717 | 0.639 | **0.779** | 0.631 | **0.865** |
| SVC (RBF) | 0.805 | 0.676 | 0.796 | 0.700 | 0.784 |

**Top 5 features by importance (Random Forest):** alcohol (0.224), sulphates
(0.142), volatile acidity (0.107), citric acid (0.097), total sulfur dioxide
(0.076) — consistent with the correlation analysis, where `alcohol` (+0.48) and
`volatile acidity` (−0.40) were the strongest linear correlates of quality.

### Reading the results honestly

Accuracy and macro F1 both favour **Random Forest**, but it comes with a real
trade-off worth flagging: it only catches **35% of actually-good wines**
(recall on the "good" class), meaning it plays it safe and defaults to "bad"
more often than the other two models. **SGD**, despite the lowest overall
accuracy, catches **87% of good wines** — at the cost of a lot of false
positives (bad wines misclassified as good), which is why its precision and
accuracy are both low. **SVC** sits in between on every metric.

Which trade-off is "right" depends on the business cost of each error type: if
missing a genuinely good wine is expensive (e.g. underpricing a premium batch),
SGD's high recall may actually be more useful despite its lower accuracy;
if false "good" flags are the costly mistake (e.g. wasted premium-tier
promotion), Random Forest's precision and overall balance win out.

## Conclusion & Recommendation

**Random Forest** is the recommended model for deployment: best accuracy and
macro F1, robust to the skewed feature distributions without needing scaling,
and — crucially for a production setting — it produces the interpretable
feature-importance ranking above, which a QA or production team can act on
directly (e.g. "alcohol and sulphates matter most").

If the priority shifts toward **not missing any good wines** rather than
overall balance, revisit this recommendation in favour of SGD or a
recall-weighted/threshold-tuned version of Random Forest. SVC is a reasonable
middle ground but offers no efficiency or interpretability advantage over
Random Forest here to justify its higher training cost.

## Limitations & Next Steps

- No hyperparameter tuning was performed (`GridSearchCV` / `RandomizedSearchCV`
  would likely improve all three models, especially SVC)
- Evaluation is a single stratified train/test split; k-fold cross-validation
  would give more robust estimates given the small dataset (1,359 rows after
  deduplication)
- The `quality_class` (3-class) target is engineered but not modelled — swap
  `y = df['quality_label']` for `y = df['quality_class']` in the notebook to
  extend the same pipeline to it
- Only the **red** wine dataset was used; the companion white wine dataset
  (`winequality-white.csv`, same schema) could be added for a larger, combined
  analysis
- Decision threshold (currently the default 0.5) could be tuned against
  `predict_proba` output to explicitly trade off precision vs. recall on the
  "good" class, rather than picking a single "best" model

## Citation

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine
> Quality [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C56S3T
