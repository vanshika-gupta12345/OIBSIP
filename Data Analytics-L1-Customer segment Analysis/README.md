# Customer Segmentation Analysis — RFM + K-Means Clustering

Segmenting an e-commerce company's customer base into distinct behavioural groups using
RFM (Recency, Frequency, Monetary) analysis and K-Means clustering — with a concrete
marketing action recommended for each segment.

> 📌 Internship project — full pipeline from raw transaction data to actionable customer
> segments.

## 📊 Project Overview

This project answers:

- Who are this business's most valuable customers, and who's at risk of churning?
- What behavioural features best describe a customer for segmentation purposes?
- How many distinct customer groups actually exist in the data (not assumed upfront)?
- What should the marketing team *do differently* for each group?

## 🗂️ Dataset

**Source:** [UCI "Online Retail" dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) —
real transaction-level data from a UK-based, non-store online retailer selling all-occasion
gifts.

| | |
|---|---|
| Rows | 541,909 order line items |
| Date range | 1 Dec 2010 – 9 Dec 2011 |
| Unique customers (after cleaning) | 4,338 |
| Columns | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

**Citation:** Chen, D., Sain, S.L., and Guo, K. (2012). *Data mining for the online retail
industry: A case study of RFM model-based customer segmentation using data mining.* Journal
of Database Marketing & Customer Strategy Management, 19(3), 197–208.

> ⚠️ **This dataset is genuinely messy — that's real, not a teaching artifact.** ~24.9% of
> rows have no `CustomerID`, some invoices are cancellations (prefixed `"C"`), and a handful
> of rows have zero or negative prices (adjustments/write-offs). No synthetic data was added
> anywhere in this project — the cleaning steps in the notebook are a genuine part of the
> analysis, not cosmetic.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`KMeans`, `StandardScaler`)
- matplotlib, seaborn
- Jupyter Notebook

## 📁 Repository Structure

```
├── README.md                               # This file
├── Customer_Segmentation_RFM_KMeans.ipynb  # Full analysis notebook (executed, with outputs)
├── online_retail_dataset.csv.gz            # Raw dataset used (uncleaned, gzip-compressed, ~7.6MB)
└── requirements.txt                        # Python dependencies
```

> The dataset is shipped gzip-compressed to stay under GitHub's 25MB web-upload limit — it's
> the exact same 541,909 rows, just compressed. `pandas.read_csv()` reads `.csv.gz` files
> directly with no extra step: `pd.read_csv("online_retail_dataset.csv.gz")`.

## 🔍 What's Inside the Notebook

1. **Initial inspection** — shape, dtypes, null-value audit
2. **Cleaning messy real-world data** — dropping unattributable transactions (missing
   `CustomerID`), cancelled orders, and invalid prices/quantities, with reasoning for each
3. **Descriptive statistics** — average order value, purchase frequency, historical customer
   lifetime value
4. **RFM feature selection** — Recency, Frequency, Monetary, with rationale for why these
   three
5. **Standardisation** — log-transforming the skewed Frequency/Monetary features, then
   `StandardScaler`, before clustering
6. **K-Means + Elbow Method** — inertia plotted for K=1–10, with the elbow at **K=4**
7. **Cluster visualisation** — two scatter plots (Recency×Monetary, Frequency×Monetary)
   coloured by cluster
8. **Cluster profiling** — mean R/F/M per cluster, translated into 4 named customer types
9. **Customers-per-cluster bar chart**
10. **Insights & recommendations** — a specific marketing action for each segment

Every code cell is pre-executed, so charts render immediately on GitHub/nbviewer with no
need to re-run anything.

## 💡 Key Findings

Four segments emerged from the elbow-selected K=4 clustering:

| Segment | Recency | Frequency | Monetary | Share of customers |
|---|---|---|---|---|
| **Champions** | ~20 days | ~16 orders | ~$10,000 | ~13% |
| **Loyal / Regular** | ~45 days | ~4 orders | ~$1,700 | ~33% |
| **New / Occasional** | ~58 days | ~1–2 orders | ~$390 | ~32% |
| **At Risk / Lapsed** | ~260 days | ~1–2 orders | ~$390 | ~22% |

- Champions are the smallest group but generate a disproportionate share of revenue per
  customer — the classic 80/20 pattern.
- Nearly a quarter of the customer base (At Risk/Lapsed) hasn't purchased in the better
  part of a year.

## ✅ Recommendations

1. **Champions:** Launch a VIP/loyalty tier with early access and exclusive perks; ask for
   reviews and referrals rather than leading with discounts, which erode margin on
   customers who'd buy anyway.
2. **Loyal/Regular:** Cross-sell and upsell based on past purchase categories, with a
   "spend more, unlock Champions perks" nudge to move them up a tier.
3. **New/Occasional:** A time-limited second-purchase incentive plus onboarding content to
   build the buying habit before they go cold.
4. **At Risk/Lapsed:** A win-back campaign with a meaningful incentive; move to low-cost,
   low-frequency contact if unresponsive rather than continuing full acquisition-level spend.

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Customer_Segmentation_RFM_KMeans.ipynb
```

## 📄 License

Dataset is provided by the UCI Machine Learning Repository for academic/educational use.
Code in this repository is shared for portfolio and educational purposes.
