# Customer Review Sentiment Analysis: VADER vs a Supervised Classifier

Sentiment analysis of Amazon consumer-electronics reviews, comparing a rule-based lexicon (VADER) with a supervised TF-IDF + Logistic Regression classifier. Both are evaluated against customers' star ratings, and the better model is used to identify what drives negative reviews and turn that into business recommendations.

**Stack:** Python · pandas · NLTK (VADER) · scikit-learn · langdetect · WordCloud · matplotlib · seaborn

---

## Data quality audit

The raw file contained **25,000 rows**, but an audit showed most were not genuine reviews:

- **5,000 synthetic placeholder rows** ("Sample Product") with generated text such as "very good" or "terrible", and impossible ratings of 0 and 2.5 stars
- **Around 17,000 duplicated reviews**, copies of real reviews with a phrase such as "Excellent product." or "Could improve." appended

These were removed, leaving **2,847 genuine reviews**. After keeping English-only reviews (VADER's lexicon is English), **2,760 reviews across 168 products** remained. Without this step, near-identical copies would have appeared in both training and test data and inflated model accuracy.

## Approach

- **Labels** from star ratings: 1–2 negative, 3 neutral, 4–5 positive
- **EDA:** rating distribution, review length, and word clouds per sentiment (lemmatised, with negations such as "not" preserved)
- **VADER** with standard thresholds (compound ≥ 0.05 positive, ≤ −0.05 negative)
- **TF-IDF (words and two-word phrases) + Logistic Regression** with balanced class weights, tuned with 5-fold cross-validation on the training set
- Both models evaluated on the same held-out 20% test set

## Results

| Model | Accuracy | Macro F1 | Negative recall |
|---|---|---|---|
| VADER (rule-based) | 0.60 | 0.43 | 0.44 |
| **TF-IDF + Logistic Regression** | **0.76** | **0.70** | **0.51** |

- VADER struggles with mixed reviews: 2-star reviews scored positive on average, and most 3-star reviews were labelled positive.
- **"Customer service"** is the most common phrase in negative reviews, alongside "not work" and "stopped working".
- **Charging accessories** (chargers, cables, power banks) make up four of the ten products with the highest share of negative reviews.

## Recommendations

1. Automated review monitoring with alerts when a product's negative share rises
2. Quality escalation with suppliers when failure phrases cluster on a product
3. Faster after-sales support for reviews mentioning returns or customer service
4. Setup guides and FAQs for smart devices with app and connectivity complaints

## Limitations

- Small negative class (317 reviews) and star ratings used as a proxy for true sentiment
- The data over-samples critical reviews and excludes non-English reviews
- A fine-tuned transformer or aspect-based sentiment analysis would be the next step

## Dataset

A publicly available dataset of Amazon consumer-electronics reviews (2024 products), obtained through an open-data repository search. The exact source page was not recorded, so the data file is not included in this repository.

## How to run

1. Open `customer_review_sentiment_analysis.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload the dataset CSV to the session as `Final_UpdatedProductReviews_withRatings.csv`
3. Choose **Runtime → Run all**

## Author

**Oladiti Abdulahi**, MSc Data Science, University of Salford
[LinkedIn](https://www.linkedin.com/in/oladiti-abdulahi-6925311a9) · [GitHub](https://github.com/Oladiti-A)
