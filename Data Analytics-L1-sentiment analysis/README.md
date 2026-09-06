# Sentiment Analysis — US Airline Tweets

A machine learning pipeline that classifies tweet sentiment (positive, negative, or neutral)
using TF-IDF features and two classifiers — with full evaluation, word-cloud visualisation,
and error analysis on real misclassified examples.

> 📌 Internship project — full NLP pipeline from raw tweets to a business-ready sentiment
> classifier.

## 📊 Project Overview

This project answers:

- What does public sentiment toward major US airlines actually look like on Twitter?
- Can a simple bag-of-words model reliably separate positive, negative, and neutral tweets —
  and where exactly does it break down?
- Naive Bayes vs. Logistic Regression: which one actually deserves to be called "better" on
  a dataset this imbalanced?

## 🗂️ Dataset

**Source:** [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) —
real tweets aimed at major US airlines, scraped in February 2015 and hand-labeled by human
annotators.

| | |
|---|---|
| Total tweets | 14,640 |
| Negative | 9,178 (62.7%) |
| Neutral | 3,099 (21.2%) |
| Positive | 2,363 (16.1%) |

This is real, messy tweet text — `@mentions`, hashtags, typos, HTML entity leftovers
(`&amp;`), and even a known artifact in the original 2015 data ("Cancelled Flightled")
that isn't a cleaning bug, it's genuinely in the raw tweets.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, metrics)
- NLTK (stopwords, tokenisation, lemmatisation)
- WordCloud
- matplotlib, seaborn
- Jupyter Notebook

## 📁 Repository Structure

```
├── README.md                                  # This file
├── Sentiment_Analysis_Airline_Tweets.ipynb    # Full analysis notebook (executed, with outputs)
├── airline_tweets_dataset.csv                 # Raw dataset used
└── requirements.txt                           # Python dependencies
```

## 🔍 What's Inside the Notebook

1. **Class distribution** — bar chart of positive/negative/neutral counts, and why the
   62/21/16 imbalance makes raw accuracy a misleading scoreboard on its own
2. **Text preprocessing pipeline** — HTML-entity decoding, lowercasing, `@mention`/URL
   stripping, punctuation removal, tokenisation, stopword removal, lemmatisation — each step
   justified in its own markdown cell
3. **TF-IDF feature extraction** — with an explanation of what it does and why it's the
   right tool here (unigrams + bigrams, 5,000 features)
4. **80/20 stratified train/test split**
5. **Two classifiers trained**: Multinomial Naive Bayes and Logistic Regression
   (`class_weight='balanced'` to counteract the imbalance)
6. **Full evaluation** — accuracy, macro precision/recall/F1, and a confusion matrix for
   each model, side by side
7. **Word clouds for each sentiment class**, plus a discussion of why "flight" and "time"
   dominate all three (they're the topic, not the sentiment)
8. **Error analysis** — 5 real misclassified tweets, each with a specific discussion of
   *why* the model got it wrong (not generic categories)
9. **Conclusion** — which model wins and why, plus a real-world application

Every code cell is pre-executed, so every chart, table, and metric renders immediately on
GitHub/nbviewer with no need to re-run anything.

## 💡 Key Findings

| Model | Accuracy | Macro F1 | Neutral Recall | Positive Recall |
|---|---|---|---|---|
| Naive Bayes | 73.9% | 0.59 | 0.27 | 0.40 |
| **Logistic Regression** | **75.4%** | **0.71** | **0.71** | **0.70** |

- **Accuracy alone is misleading here.** A model that always predicted "negative" would
  already score ~63% without learning anything.
- **Naive Bayes is biased toward the majority class** — it barely beats Logistic Regression
  on accuracy, but its recall on neutral (0.27) and positive (0.40) tweets is far worse,
  because it wasn't corrected for the 62/21/16 imbalance.
- **Logistic Regression, trained with `class_weight='balanced'`, wins clearly** on macro F1
  (0.71 vs. 0.59) — the metric that actually reflects performance across all three classes,
  not just the biggest one.
- The 5 misclassified examples analysed in the notebook show the model mostly fails when a
  single strongly-charged word ("Cancelled", "awesome") appears in a tweet that isn't
  actually expressing sentiment about the airline itself.

## ✅ Conclusion

**Best model: Logistic Regression**, judged on macro F1 rather than raw accuracy, since this
dataset's class imbalance makes accuracy an unreliable single metric.

**Real-world application:** automated **customer service triage and brand monitoring** —
scanning incoming tweets in real time, routing negative tweets (especially ones mentioning
concrete pain points) to urgent human follow-up, sending neutral questions to a standard
support queue, and surfacing positive tweets for marketing use. The same pipeline transfers
directly to product reviews, app store feedback, or support-ticket triage in any industry.

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
jupyter notebook Sentiment_Analysis_Airline_Tweets.ipynb
```

## 📄 License

Dataset originally collected by Crowdflower/Figure Eight and distributed on Kaggle for
educational use. Code in this repository is shared for portfolio and educational purposes.
