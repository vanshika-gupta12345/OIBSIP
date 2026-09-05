# Retail Sales — Exploratory Data Analysis

Exploratory Data Analysis on a real-world retail sales dataset, uncovering revenue trends,
customer behaviour patterns, and product performance — with actionable recommendations for
the business.

> 📌 Internship project — full EDA workflow from raw data to business recommendations.

## 📊 Project Overview

This project analyzes ~4 years of order-level transactions from a US retail chain to answer:

- How do sales trend over time — is the business growing, and is it seasonal?
- Who are the customers, and does age/gender relate to buying behaviour?
- Which products and categories drive volume vs. revenue?
- How do sales, quantity, discount, and profit relate to each other?
- Where is the business quietly losing money, and what should it do differently?

## 🗂️ Dataset

**Source:** [Sample Superstore dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — a widely-used, freely available retail dataset (the standard result for "superstore sales" on Kaggle).

| | |
|---|---|
| Rows | 9,994 order line items |
| Date range | Jan 2014 – Dec 2017 |
| Unique customers | 793 |
| Unique products | 1,850 |
| Categories | Furniture, Office Supplies, Technology |

**Columns:** Order ID/date, Ship date, Ship Mode, Customer ID/Name, Segment, geography
(Country/State/City/Region/Postal Code), product hierarchy (Category → Sub-Category →
Product Name), and the core metrics `Sales`, `Quantity`, `Discount`, `Profit`.

> ⚠️ **Note on demographics:** Real transaction logs (this one included) don't contain
> personal fields like age or gender — retailers don't collect or publish that at the
> line-item level. To cover the customer-demographics part of this analysis, two columns
> — `Customer Age` and `Gender` — were **simulated** with a fixed random seed, assigned
> consistently per unique customer. Every other field is the original, real dataset. This is
> flagged again at the top of the notebook so it's never mistaken for real customer data.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook

## 📁 Repository Structure

```
├── README.md                    # This file
├── Retail_Sales_EDA.ipynb       # Full analysis notebook (executed, with outputs)
├── retail_sales_dataset.csv     # Dataset used (original + simulated demographic columns)
└── requirements.txt             # Python dependencies
```

## 🔍 What's Inside the Notebook

1. **Initial inspection** — shape, dtypes, null-value audit (uncovers a real ~2% block of
   incomplete rows and how they're handled)
2. **Descriptive statistics** — mean, median, mode, std for all numerical columns
3. **Time series analysis** — monthly and quarterly sales trend lines
4. **Customer demographics** — age-group distribution, gender breakdown
5. **Product analysis** — top 10 best-selling products by units sold; revenue by category
6. **Correlation heatmap** — relationships between Sales, Quantity, Discount, Profit, Age
7. **Extra insight** — average profit by discount band, split by category (reveals where
   discounting turns unprofitable)
8. **Conclusion** — 4 specific, data-backed business recommendations

Every chart is followed by a short written observation, and every code cell is pre-executed
so charts render immediately on GitHub/nbviewer without needing to re-run anything.

## 💡 Key Findings

- Sales grow year-over-year but are sharply seasonal — Q4 (especially November) peaks every
  year, Q1 dips every year.
- Office Supplies drives the most *units sold*; Technology drives the most *revenue* despite
  fewer transactions — two different growth levers.
- Discounts above ~20% flip average profit **negative** in Furniture and Office Supplies,
  but not in Technology.
- The customer base skews 25–54 years old with a near-even gender split.

## ✅ Recommendations

1. Cap discretionary discounts on Furniture and Office Supplies at ~20%; give Technology
   more discounting room since it holds profit better at depth.
2. Run a Q1 demand-smoothing campaign to flatten the seasonal trough.
3. Track Office Supplies on retention/reorder rate and Technology on average order value —
   don't manage both against the same KPI.
4. Turn the top-10 best-selling (low-cost, repeat-purchase) products into a
   reorder-reminder or auto-replenishment offer.

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Retail_Sales_EDA.ipynb
```

## 📄 License

Dataset is fictional and provided for academic/educational use. Code in this repository is
shared for portfolio and educational purposes.
