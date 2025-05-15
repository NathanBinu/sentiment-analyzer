# scripts/grid_search.py

import os
import sys
import pandas as pd
import joblib
import nltk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ────────────────────────────────────────────────────────────────
# 1) Locate project root & make sure we can import your preprocess code
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess_and_train import clean_text, RAW_CSV  # noqa: E402

# 2) Prepare VADER
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# ────────────────────────────────────────────────────────────────

def load_and_label():
    # Load
    df = pd.read_csv(RAW_CSV)

    # Clean
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # VADER label
    df["compound"] = df["text"].fillna("").apply(lambda t: sia.polarity_scores(t)["compound"])
    df["label"] = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    return df

def main():
    df = load_and_label()
    X = df["cleaned"]
    y = df["label"]

    # 3) Pipeline & Parameter Grid
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",  LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
        "tfidf__max_df": [0.7, 0.8, 0.9],
        "tfidf__min_df": [3, 5, 10],
        "clf__C": [0.1, 1, 10],
    }

    # 4) GridSearchCV
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)

    print("\n🔍 Best parameters:", grid.best_params_)
    print("⭐  Best CV f1_weighted score:", grid.best_score_)

    # 5) Save the best model
    out_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    joblib.dump(grid.best_estimator_, out_path)
    print(f"✅ Tuned model saved to {out_path}")

    # 6) Honest test‐set evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = grid.best_estimator_.predict(X_test)
    print("\n🔎 Test‐set performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
