"""
train_model.py

Trains a Fake News classifier (TF-IDF + Logistic Regression) and saves:
  - models/fake_news_pipeline.joblib   (vectorizer + classifier in one pipeline)
  - models/metrics.json                (accuracy/precision/recall/F1)

Logistic Regression is used deliberately (not a black-box deep model)
because it plays extremely well with LIME/SHAP explanations while still
giving strong accuracy on this task -- ideal for an "Explainable AI"
project where the explanation quality matters as much as raw accuracy.

USAGE
-----
1) Using the built-in tiny sample dataset (for testing the pipeline runs):
     python data/make_sample_data.py
     python train_model.py --data data/sample_news.csv --text_col text --label_col label

2) Using the Kaggle "Fake and Real News Dataset" (Fake.csv + True.csv):
     python train_model.py --fake data/Fake.csv --real data/True.csv

3) Using any custom CSV with a text column and a label column:
     python train_model.py --data data/mydata.csv --text_col text --label_col label
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_kaggle_style(fake_path: str, real_path: str) -> pd.DataFrame:
    """Loads the common Kaggle 'Fake and Real News' Fake.csv / True.csv format."""
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # This dataset usually has 'title' and 'text' columns -- combine them.
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")

    return df[["text", "label"]]


def load_generic_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]


def build_pipeline() -> Pipeline:
    """
    TfidfVectorizer + LogisticRegression wrapped in a single sklearn Pipeline.
    Wrapping in a Pipeline means LIME can call `.predict_proba` on raw text
    directly, with no manual vectorizing step needed at explanation time.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
        )),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str, help="Path to Fake.csv (Kaggle-style)")
    parser.add_argument("--real", type=str, help="Path to True.csv (Kaggle-style)")
    parser.add_argument("--data", type=str, help="Path to a generic CSV with text+label columns")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.fake and args.real:
        df = load_kaggle_style(args.fake, args.real)
    elif args.data:
        df = load_generic_csv(args.data, args.text_col, args.label_col)
    else:
        raise SystemExit(
            "Provide either --fake and --real (Kaggle-style) or --data (generic CSV). "
            "For a quick test run: python data/make_sample_data.py "
            "then: python train_model.py --data data/sample_news.csv"
        )

    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=args.test_size,
        random_state=42, stratify=df["label"]
    )

    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    print(report)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": sorted(df["label"].unique().tolist()),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")
    print(f"Saved metrics to {os.path.join(MODELS_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
