# NLP-Based Sentiment Analysis of Amazon Product Reviews

A Mini NLP Project that classifies Amazon product reviews as **positive** or
**negative** using TF-IDF feature extraction and a Logistic Regression
classifier.

## Problem
Online shoppers and sellers deal with huge volumes of product reviews.
Reading every review manually to gauge sentiment doesn't scale. This
project automatically classifies review text as positive or negative.

## Dataset
[Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset)
from Kaggle — 4,915 reviews with review text and star ratings (1–5).
After cleaning (removing missing/duplicate/neutral reviews) and labeling
by rating, the final dataset has 4,770 reviews (4,446 positive, 324
negative). See `report/report.md` for full details, including an
important discussion of dataset imbalance.

## Methodology
```
Raw Reviews -> Cleaning & Sentiment Labeling (from star rating)
            -> Text Preprocessing (stopwords removed, negations kept)
            -> Train/Test Split (80/20)
            -> TF-IDF (unigrams + bigrams, 5000 features)
            -> Logistic Regression
            -> Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

## Results
- **Accuracy:** 94.44%
- **Positive class:** Precision 0.94, Recall 1.00, F1 0.97
- **Negative class:** Precision 0.93, Recall 0.20, F1 0.33

The high accuracy is misleading on its own — the dataset is ~93% positive,
so the model struggles significantly to detect negative reviews. See
`report/report.md` for the full discussion.

## How to Run
```bash
pip install -r requirements.txt

# Set up Kaggle API credentials first (kaggle.json in ~/.kaggle/)
kaggle datasets download -d gzdekzlkaya/amazon-product-reviews-dataset -p dataset --unzip

python source_code.py
```

## Project Structure
```
NLP-Project/
│
├── README.md
├── source_code.py
├── dataset/
│   └── amazon_review.csv
├── screenshots/
│   ├── confusion_matrix.png
│   └── class_distribution.png
├── report/
│   └── report.md
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── requirements.txt
```

## Author
[Your Name]
