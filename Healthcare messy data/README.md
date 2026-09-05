# Data Cleaning: Messy Healthcare Dataset

Taking a deliberately messy, real-world-inspired dataset and systematically transforming it
into a clean, analysis-ready one — with every cleaning decision documented and justified,
not just applied.

> 📌 Internship project — full data quality audit, cleaning pipeline, and before/after
> validation.

## 📊 Project Overview

This project demonstrates professional-level data cleaning on a dataset engineered to
contain realistic problems:

- How do you audit a dataset for quality issues before touching it?
- When should you delete a row, impute a value, or leave it explicitly unknown — and how do
  you justify that choice per column rather than applying one rule to everything?
- How do you standardise inconsistent categories and multiple date formats without silently
  destroying information?
- How do you tell a genuine statistical outlier apart from a data-entry sentinel value like
  `999`?

## 🗂️ Dataset

**Source:** [Messy Healthcare Dataset](https://github.com/DanEinstein/Data_Analysis) — a
dataset deliberately engineered by its original author to simulate real-world data quality
problems for cleaning practice.

| | |
|---|---|
| Rows (raw) | 203 patient records |
| Columns | 14 (ID, name, age, gender, department, diagnosis, admit date, doctor, insurance, status, billing amount, phone, email) |
| Rows (cleaned) | 195 |

**Known issues in the raw file** (all genuinely present, not added for this project):
mixed-case/abbreviated categorical values (`male`/`Male`/`M`), five different date formats
mixed in one column, a billing amount column stored as text due to stray `"$"` prefixes,
impossible age values (`-1`, `999`), 3 exact duplicate rows, and — the trickiest one — 7 rows
where two *different* patients were accidentally given the *same* ID.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- Jupyter Notebook

## 📁 Repository Structure

```
├── README.md                                  # This file
├── Data_Cleaning_Healthcare_Dataset.ipynb      # Full cleaning notebook (executed, with outputs)
├── healthcare_dataset_messy.csv                # Original raw dataset (untouched)
├── healthcare_dataset_cleaned.csv              # Final cleaned output
└── requirements.txt                            # Python dependencies
```

## 🔍 What's Inside the Notebook

1. **Data quality report** — nulls per column, duplicate rows, dtype issues, value-range
   anomalies, category inconsistencies, and date-format chaos, all surfaced before any
   cleaning starts
2. **Duplicate removal** — 3 exact duplicates dropped; 7 ID-collision rows (same ID,
   different person) disambiguated instead of deleted, since deleting either would discard a
   real patient's record
3. **Type correction & standardisation** — `billing_amount` → float, `admit_date` parsed
   across 5 formats → datetime, categorical text unified (`"ER"`→`"Emergency"`,
   `"self-pay"`→`"Self Pay"`, etc.), phone numbers reformatted, emails lowercased
4. **Outlier/anomaly detection** — impossible age sentinels (`-1`, `999`) separated from
   genuine statistical outliers (IQR + Z-score on `billing_amount` and cleaned `age`)
5. **Missing-value handling** — a *different, justified* strategy per column: median
   imputation, mode imputation (flagged for transparency), row deletion, and explicit
   "Unknown"/"Not Recorded"/"Not Provided" categories where guessing would be inappropriate
6. **Final dtype correction** — IDs as string, age as nullable integer, categoricals as
   `category` dtype
7. **Before vs. after summary table** — row count, null count, duplicate count, dtype
   accuracy, side by side
8. **Cleaned CSV saved** to disk

Every code cell is pre-executed, so the full audit trail — including every intermediate
table — renders immediately on GitHub/nbviewer.

## 💡 Key Decisions Worth Knowing

- **Row deletion was used exactly once, deliberately** — for the 5 rows (2.5%) with an
  unrecoverable `billing_amount`. A financial figure is too consequential to guess at, and
  the sample-size cost was small.
- **Mode imputation was used exactly once, and flagged** — for `department` only, with an
  `department_was_imputed` column so anyone doing department-level analysis can exclude
  those rows if they want.
- **Four columns deliberately avoid mode imputation** (`gender`, `diagnosis`,
  `attending_doctor`, `insurance_provider`, `status`) in favour of an explicit "unknown"
  category, because guessing a specific fact about a specific patient (their gender, their
  diagnosis, who treated them) is a different — and worse — kind of error than leaving it
  visibly unresolved.
- **The `"ER"` → `"Emergency"` merge is flagged as a judgment call**, not silently applied —
  a real analyst would confirm this kind of category merge with the source system owner.

## 📈 Before vs. After

| Metric | Before | After |
|---|---|---|
| Row count | 203 | 195 |
| Total null values | 207 | 0 |
| Exact duplicate rows | 3 | 0 |
| `age` dtype | `float64` (contaminated by sentinels) | `Int64` (nullable, sentinel-free) |
| `billing_amount` dtype | `object` (`$` prefix on some rows) | `float64` |
| `admit_date` dtype | `object` (5 mixed formats) | `datetime64[ns]` |
| `patient_id` dtype | `object` | `string`, collisions disambiguated |

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Data_Cleaning_Healthcare_Dataset.ipynb
```

## 📄 License

Dataset is provided by its original author for educational/data-cleaning-practice use. Code
in this repository is shared for portfolio and educational purposes.
