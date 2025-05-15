import re
import joblib
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score  # ← added

# Paths
RAW_CSV    = "data/raw_combined.csv"
MODEL_PATH = "models/baseline.pkl"
VECT_PATH  = "models/vectorizer.pkl"

# Ensuring VADER is ready to process
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


def clean_text(text):
    "Simple cleanup: remove URLs, mentions, non-alphanumeric, lowercase."
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    return text.lower().strip()


def main():
    # 1. Load
    df = pd.read_csv(RAW_CSV)
    if df.empty:
        raise RuntimeError(f"No data found in {RAW_CSV}! Run fetch_reddit.py first.")

    # 2. Data Cleanup
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # 3. Labels via VADER
    df["compound"] = df["text"].fillna("").apply(lambda t: sia.polarity_scores(t)["compound"])
    df["label"] = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    # 4. Balancing classes by up-sampling
    neg = df[df.label == 0]
    pos = df[df.label == 1]
    neu = df[df.label == 2]

    max_count = max(len(neg), len(pos), len(neu))

    neg_up = neg.sample(max_count, replace=True, random_state=42)
    pos_up = pos.sample(max_count, replace=True, random_state=42)
    neu_up = neu.sample(max_count, replace=True, random_state=42)

    df = pd.concat([neg_up, pos_up, neu_up]).sample(frac=1, random_state=42)
    print("Balanced class distribution:\n", df.label.value_counts(), "\n")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.2, random_state=42
    )

    # 6. Vectorize & Train
    vect  = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    Xtr   = vect.fit_transform(X_train)
    Xte   = vect.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)

    # 7. Evaluate: accuracy + classification report
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Vectorizer saved to {VECT_PATH}")


if __name__ == "__main__":
    main()
