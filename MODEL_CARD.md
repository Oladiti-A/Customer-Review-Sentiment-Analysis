# Model Card: Customer Review Sentiment Classifier

**Author:** Oladiti Abdulahi · **Version:** 1.0 (September 2026) · **Notebook:** `customer_review_sentiment_analysis.ipynb`

## Model details

| | |
|---|---|
| **Task** | Three-class sentiment classification of product reviews: negative, neutral, positive |
| **Model** | TF-IDF features (words and two-word phrases, `min_df=2`, sublinear TF) + Logistic Regression (`class_weight="balanced"`) |
| **Tuning** | Regularisation strength `C` selected by 5-fold stratified cross-validation on the training set, scored by macro F1 (best `C = 1`) |
| **Baseline** | NLTK VADER lexicon with standard thresholds (compound ≥ 0.05 positive, ≤ −0.05 negative) |
| **Framework** | Python, scikit-learn, NLTK |

## Intended use

- **Intended:** monitoring customer reviews of consumer electronics at product level, flagging products with rising negative feedback, and identifying common complaint themes for quality and customer service teams.
- **Users:** analysts and product or customer service managers, with a person reviewing flagged results.
- **Out of scope:** decisions about individual customers or reviewers; non-English text; domains far from consumer electronics (e.g. healthcare or finance) without retraining; detecting fake reviews or sarcasm.

## Training and evaluation data

- **Source:** a publicly available dataset of Amazon consumer-electronics reviews (2024 products). The exact source page was not recorded.
- **Data audit:** the raw file had 25,000 rows, but 5,000 were synthetic placeholder rows and about 17,000 were duplicated reviews with short phrases appended. These were removed, leaving 2,847 genuine reviews.
- **Filtering:** English-only reviews kept (language detected with `langdetect`), giving **2,760 reviews across 168 products**.
- **Labels:** derived from star ratings: 1–2 negative (317), 3 neutral (918), 4–5 positive (1,525).
- **Split:** 80/20 stratified split, 2,208 training and 552 test reviews. Deduplication ensured no review appears in both sets.

## Metrics

**Primary metric: macro F1**, which averages performance across the three classes equally, so the model cannot score well by only predicting the large positive class. Accuracy and negative-class recall are also reported, because missing unhappy customers is the most costly error for the business.

## Results (held-out test set, 552 reviews)

| Model | Accuracy | Macro F1 | Negative recall |
|---|---|---|---|
| VADER (baseline) | 0.60 | 0.43 | 0.44 |
| **TF-IDF + Logistic Regression** | **0.76** | **0.70** | **0.51** |

Per-class F1 for the classifier: negative 0.58, neutral 0.67, positive 0.84.

## Limitations

- **Small negative class (317 reviews):** negative predictions are the weakest; about half of negative reviews are still missed.
- **Labels are a proxy:** star ratings do not always match the text (e.g. a 3-star review written positively).
- **Representativeness:** the data over-samples critical reviews and excludes non-English reviews, so results may not reflect all customers.
- **Bag-of-words model:** it does not understand context, so negation, sarcasm and mixed reviews ("great screen, but it stopped charging") can be misread.
- **Single random split:** results come from one 80/20 split; repeated splits would give confidence intervals.

## Ethical considerations

- Reviews are public but were written for other shoppers. No reviewer names or identifiers are used, and any production use should follow GDPR and the platform's terms of service.
- Excluding non-English reviews risks overlooking problems affecting international customers.
- Outputs should be used to improve products and service, never to target or suppress individual reviewers.

## Recommendations and next steps

- Use for product-level trends with human review, not automated decisions about individuals.
- Retrain regularly on the company's own labelled reviews.
- Next improvements: a fine-tuned transformer (e.g. DistilBERT) to handle context and negation, aspect-based sentiment (battery, delivery, customer service), and a multilingual model.
