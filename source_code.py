"""
source_code.py
----------------
NLP-Based Sentiment Analysis of Amazon Product Reviews
Using TF-IDF and Logistic Regression

Dataset: Amazon Product Reviews Dataset (Kaggle)
https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset

Pipeline:
Raw Reviews -> Cleaning -> Sentiment Labeling (from star rating)
            -> Text Preprocessing -> Train/Test Split
            -> TF-IDF -> Logistic Regression -> Evaluation -> Prediction

This script reproduces, end-to-end, the exact steps run in Google Colab.
"""

import os
import re
import string
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("report", exist_ok=True)

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
NEGATION_WORDS = {"not", "no", "nor", "never", "n't"}
STOP_WORDS = STOP_WORDS - NEGATION_WORDS  # keep negations, they flip sentiment


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
def load_data(path="dataset/amazon_review.csv"):
    df = pd.read_csv(path)
    print("Loaded shape:", df.shape)
    print("Rating distribution:\n", df["overall"].value_counts().sort_index())
    return df


# ---------------------------------------------------------
# 2. CLEAN + LABEL
# ---------------------------------------------------------
def clean_and_label(df):
    before = len(df)
    df = df.dropna(subset=["reviewText"])
    df = df.drop_duplicates(subset="reviewText")
    df = df[df["overall"] != 3.0]  # drop neutral

    df = df.copy()
    df["sentiment"] = df["overall"].apply(lambda r: "positive" if r >= 4 else "negative")

    print(f"Removed {before - len(df)} rows (missing/duplicate/neutral)")
    print("Final dataset size:", df.shape)
    print(df["sentiment"].value_counts())
    return df


# ---------------------------------------------------------
# 3. TEXT PREPROCESSING
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def preprocess(df):
    df = df.copy()
    df["clean_review"] = df["reviewText"].apply(clean_text)
    before = len(df)
    df = df[df["clean_review"].str.strip() != ""]
    print(f"Removed {before - len(df)} empty reviews after cleaning")
    return df


# ---------------------------------------------------------
# 4. TRAIN/TEST SPLIT + TF-IDF
# ---------------------------------------------------------
def split_and_vectorize(df):
    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training TF-IDF shape:", X_train_tfidf.shape)
    print("Testing TF-IDF shape:", X_test_tfidf.shape)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    print("Model trained successfully")
    return model


# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
def evaluate(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
    print("Confusion Matrix (rows=actual, cols=predicted, order=[positive, negative]):")
    print(cm)
    return cm


def plot_confusion_matrix(cm, labels=("positive", "negative")):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Logistic Regression")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: screenshots/confusion_matrix.png")


def plot_class_distribution(df):
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Distribution in Dataset")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("screenshots/class_distribution.png", dpi=150)
    plt.close()
    print("Saved: screenshots/class_distribution.png")


# ---------------------------------------------------------
# 7. SAVE MODEL + PREDICT ON NEW REVIEWS
# ---------------------------------------------------------
def save_artifacts(model, vectorizer):
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved")


def predict_samples(model, vectorizer, samples):
    cleaned = [clean_text(s) for s in samples]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    print("\n===== Sample Predictions =====")
    for text, pred in zip(samples, preds):
        print(f"Review: {text}\nPredicted Sentiment: {pred.upper()}\n")
    return preds


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    df = load_data()
    df = clean_and_label(df)
    plot_class_distribution(df)

    df = preprocess(df)

    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = split_and_vectorize(df)

    model = train_model(X_train_tfidf, y_train)

    cm = evaluate(model, X_test_tfidf, y_test)
    plot_confusion_matrix(cm)

    save_artifacts(model, vectorizer)

    sample_reviews = [
        "This product is excellent. The quality is very good and I am extremely happy with my purchase.",
        "This product is very bad. The quality is poor and I am not satisfied with my purchase.",
        "Decent product for the price, though it could be better.",
    ]
    predict_samples(model, vectorizer, sample_reviews)


if __name__ == "__main__":
    main()
