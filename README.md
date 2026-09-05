# NLP-Based Sentiment Analysis of Amazon Product Reviews Using TF-IDF and Logistic Regression

## 1. Problem Statement
Online shopping platforms accumulate thousands of customer reviews for every
product. Manually reading and interpreting these reviews to gauge overall
customer satisfaction is slow and does not scale. This project addresses the
problem of automatically classifying Amazon product reviews as **positive**
or **negative** using NLP-based text classification.

## 2. Objectives
- Collect and analyze a real-world product review dataset.
- Clean and preprocess raw review text for machine learning.
- Convert star ratings into binary sentiment labels.
- Extract numerical features from text using TF-IDF.
- Train a Logistic Regression classifier for sentiment prediction.
- Evaluate the model using standard classification metrics.
- Analyze limitations, especially the effect of class imbalance.

## 3. Problem Relevance
- **Who benefits:** E-commerce sellers, product manufacturers, and shoppers
  who want a quick read on overall customer sentiment.
- **Practical problem addressed:** Manually reading thousands of reviews to
  gauge sentiment is impractical; automated classification scales this up.
- **Why NLP is suitable:** Reviews are unstructured free text. NLP
  techniques are specifically built to extract structured signals (like
  sentiment) from such text.
- **Applications:** Customer feedback dashboards, review-based product
  ranking, and flagging negative reviews for customer support follow-up.

## 4. Dataset
- **Dataset name:** Amazon Product Reviews Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset
- **Original size:** 4,915 reviews, 12 columns
- **Key attributes used:** `reviewText` (the review text), `overall`
  (star rating, 1–5)
- **Target variable:** `sentiment` (derived from `overall`)
- **Number of classes:** 2 (binary — positive / negative)

**Original rating distribution:**

| Rating | Number of Reviews |
|--------|-------------------|
| 1      | 244               |
| 2      | 80                |
| 3      | 142               |
| 4      | 527               |
| 5      | 3,922             |

**Data cleaning performed:**
- Missing reviews removed: 1
- Duplicate reviews removed: 2
- Neutral (3-star) reviews removed: 142, since the project performs binary
  classification only
- 0 reviews became empty after text preprocessing

**Sentiment labeling rule:**
- Rating ≥ 4 → **Positive**
- Rating ≤ 2 → **Negative**
- Rating = 3 → removed (neutral)

**Final dataset:** 4,770 reviews

| Sentiment | Number of Reviews |
|-----------|-------------------|
| Positive  | 4,446             |
| Negative  | 324               |
| **Total** | **4,770**         |

**Dataset limitations:**
- The dataset is **heavily imbalanced** — about 93% of reviews are
  positive and only 7% negative. This is a real characteristic of the raw
  Kaggle data, not something introduced during cleaning, and it has a
  direct, measurable effect on model performance (see Results).
- Reviews are English-only and drawn from a single retail platform
  (Amazon), so the vocabulary may not generalize to other domains.

## 5. Methodology
```
Raw Reviews (reviewText, overall)
        |
        v
Data Cleaning (remove missing, duplicate, neutral reviews)
        |
        v
Sentiment Labeling (rating >= 4 -> positive, rating <= 2 -> negative)
        |
        v
Text Preprocessing (lowercase, remove URLs/punctuation/numbers,
                     stopword removal with negation words kept)
        |
        v
Train/Test Split (80% / 20%, stratified)
        |
        v
TF-IDF Vectorization (unigrams + bigrams, 5,000 features)
        |
        v
Logistic Regression (max_iter=1000)
        |
        v
Prediction (positive / negative)
        |
        v
Evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
```

**Stage explanations:**
1. **Cleaning & labeling:** Star ratings are converted into binary
   sentiment labels; missing, duplicate, and neutral (3-star) reviews are
   removed to keep the classification task well-defined.
2. **Preprocessing:** Text is lowercased, URLs/punctuation/numbers are
   stripped, and English stopwords are removed — **except** negation
   words (`not`, `no`, `nor`, `never`), which are kept because they can
   reverse the sentiment of a sentence (e.g. "not good" vs "good").
3. **TF-IDF:** Converts cleaned text into a 5,000-dimension numeric vector
   per review, using both unigrams and bigrams to capture short phrases.
   The vectorizer is fit only on training data and then applied to test
   data, to avoid test-set leakage.
4. **Logistic Regression:** A simple, well-understood linear classifier,
   chosen because it performs strongly on high-dimensional TF-IDF text
   features, is efficient to train, and is easy to explain — matching the
   assignment's preference for simple, understandable models.
5. **Evaluation:** Standard binary classification metrics measure how well
   the model distinguishes positive from negative reviews on unseen data.

## 6. Implementation
- **Programming language:** Python 3 (developed in Google Colab)
- **Libraries used:** `pandas`, `nltk`, `scikit-learn`, `matplotlib`,
  `joblib`, `kaggle` (for dataset download)
- **NLP techniques:** Text cleaning, stopword removal (with negation
  preservation), TF-IDF vectorization (unigrams + bigrams)
- **Algorithm/model:** Logistic Regression (`max_iter=1000, random_state=42`)
- **Implementation details:**
  - Dataset downloaded directly from Kaggle using the Kaggle API
  - `train_test_split(test_size=0.2, random_state=42, stratify=y)`
  - `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`
  - Trained model and vectorizer saved with `joblib` for reuse
    (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`)

## 7. Results

**Train/test split:**

| Set      | Number of Reviews |
|----------|-------------------|
| Training | 3,816              |
| Testing  | 954                |

**TF-IDF feature shapes:** Training (3816, 5000), Testing (954, 5000)

**Overall Accuracy: 94.44%**

**Classification Report:**

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Negative     | 0.93      | 0.20   | 0.33     | 65      |
| Positive     | 0.94      | 1.00   | 0.97     | 889     |
| **Accuracy** |           |        | **0.94** | 954     |
| Macro avg    | 0.94      | 0.60   | 0.65     | 954     |
| Weighted avg | 0.94      | 0.94   | 0.93     | 954     |

**Confusion Matrix** (rows = actual, columns = predicted; order = [positive, negative]):

|                  | Predicted Positive | Predicted Negative |
|------------------|--------------------|--------------------|
| **Actual Positive** | 888             | 1                  |
| **Actual Negative** | 52              | 13                 |

See `screenshots/confusion_matrix.png` and `screenshots/class_distribution.png`.

**Sample predictions on new, unseen reviews:**

| Review | Predicted Sentiment |
|---|---|
| "This product is excellent. The quality is very good and I am extremely happy with my purchase." | POSITIVE |
| "This product is very bad. The quality is poor and I am not satisfied with my purchase." | POSITIVE |
| "Decent product for the price, though it could be better." | POSITIVE |

The second example is a clearly negative review ("very bad," "poor,"
"not satisfied") that the model still predicted as positive — a direct,
concrete demonstration of the model's bias toward the positive class,
discussed further below.

## 8. Limitations
- **Severe class imbalance:** The dataset contains 4,446 positive reviews
  but only 324 negative reviews (~93%/7% split). This causes the model to
  strongly favor predicting "positive."
- **Very low negative recall (20%):** The model correctly identifies only
  13 out of 65 actual negative reviews in the test set; 52 negative
  reviews are misclassified as positive. This is the most significant
  weakness of the current system.
- **Misclassification of clearly negative language:** As shown in the
  sample predictions, even an explicitly negative review was predicted as
  positive, confirming the imbalance problem is a real, practical issue
  and not just a number in a table.
- **No handling of neutral/mixed sentiment:** Only binary classification
  is supported; 3-star (neutral) reviews were excluded entirely.
- **Difficulty with sarcasm and nuance:** Like most TF-IDF/bag-of-words
  approaches, the model cannot detect sarcasm or subtle mixed opinions.
- **Domain-specific:** The model is trained on Amazon product reviews and
  may not generalize well to other types of text (e.g., news, social
  media).

## 9. Future Scope
- **Address class imbalance directly:** Use class weighting
  (`class_weight="balanced"` in Logistic Regression), oversampling
  (e.g. SMOTE), or under-sampling the positive class to improve negative
  recall.
- **Collect more negative examples:** A more balanced dataset would likely
  improve the model's ability to detect negative sentiment.
- **Compare multiple models:** Naive Bayes, SVM, and Random Forest could
  be compared against Logistic Regression to find the best-performing
  option for this imbalanced setting.
- **Explore deep learning/transformer models:** LSTM, GRU, or BERT-based
  models could better capture context, negation, and sarcasm.
- **Add a neutral class:** Reintroducing 3-star reviews as a "neutral"
  category would make the system more realistic.
- **Deploy as a web application:** A simple Streamlit app could let users
  paste a review and get an instant sentiment prediction using the saved
  `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.
- **Aspect-based sentiment analysis:** Future versions could identify
  sentiment toward specific aspects (price, quality, delivery) rather than
  the review as a whole.

## 10. Conclusion
This project built a complete NLP pipeline that downloads real Amazon
product review data from Kaggle, cleans and labels it based on star
ratings, extracts TF-IDF features, and classifies reviews as positive or
negative using Logistic Regression. Starting from 4,915 raw reviews, the
dataset was cleaned down to 4,770 usable reviews (4,446 positive, 324
negative). The trained model achieved an overall accuracy of **94.44%**,
but a closer look at the classification report and confusion matrix
reveals that this accuracy is driven almost entirely by the model's strong
performance on the majority (positive) class — negative-review recall is
only **20%**. This result is a clear, practical illustration of why
accuracy alone can be a misleading metric on imbalanced datasets, and it
directly motivates future work on class balancing techniques. The project
demonstrates the complete NLP workflow — data acquisition, cleaning,
preprocessing, feature extraction, modeling, evaluation, and error

## 11. Author
Name: Elsa Elizabeth Issac
