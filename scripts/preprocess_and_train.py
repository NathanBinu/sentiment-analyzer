import re
import joblib
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score  # ← added

# File paths – adjust these if you move your data or models elsewhere
RAW_CSV    = "data/raw_combined.csv"
MODEL_PATH = "models/baseline.pkl"
VECT_PATH  = "models/vectorizer.pkl"

# Download VADER’s sentiment lexicon (if not already present) and set up the analyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


def clean_text(text):
    """
    Perform simple text cleanup:
    - Remove URLs
    - Strip out @mentions
    - Keep only letters, numbers, spaces, and apostrophes
    - Convert everything to lowercase
    """
    text = re.sub(r"http\S+", "", text)      # drop any links
    text = re.sub(r"@\w+", "", text)         # remove Twitter/Reddit mentions
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)    # strip non-alphanumeric
    return text.lower().strip()     # lowercase + trim whitespace


def main():
    # 1. Load data from CSV
    df = pd.read_csv(RAW_CSV)
    if df.empty:
        raise RuntimeError(f"No data found in {RAW_CSV}! Run fetch_reddit.py first.")

    # 2. Clean each post’s text
    #    Create a new “cleaned” column with URLs, mentions, and bad chars removed
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # 3. Auto-label via VADER’s compound score
    #    compound ≥ 0.05 → positive (label=1)
    #    compound ≤ -0.05 → negative (label=0)
    #    otherwise → neutral (label=2)
    df["compound"] = df["text"].fillna("").apply(lambda t: sia.polarity_scores(t)["compound"])
    df["label"] = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    # 4. Balance the classes by up-sampling the smaller groups
    #    This prevents the model from always predicting the majority class
    neg = df[df.label == 0]
    pos = df[df.label == 1]
    neu = df[df.label == 2]

    max_count = max(len(neg), len(pos), len(neu))  # target size for each class

    # Randomly sample with replacement to reach max_count for each class
    neg_up = neg.sample(max_count, replace=True, random_state=42)
    pos_up = pos.sample(max_count, replace=True, random_state=42)
    neu_up = neu.sample(max_count, replace=True, random_state=42)

    # Concatenate back into one DataFrame and shuffle
    df = pd.concat([neg_up, pos_up, neu_up]).sample(frac=1, random_state=42)
    print("Balanced class distribution:\n", df.label.value_counts(), "\n")

    # 5. Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.2, random_state=42
    )

    # 6. Convert text to TF–IDF features and train Logistic Regression
    vect  = TfidfVectorizer(ngram_range=(1,2), max_features=5000)   # use unigrams + bigrams
    Xtr   = vect.fit_transform(X_train)   # learn vocab & transform train
    Xte   = vect.transform(X_test)     # transform test

    model = LogisticRegression(max_iter=200)   # initialize classifier
    model.fit(Xtr, y_train)   # teach it on the training data

    # 7. Evaluate: accuracy + classification report
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")          # overall accuracy
    print(classification_report(y_test, y_pred, zero_division=0))  # precision/recall/F1 score per class

    # 8. Save the trained model and vectorizer for reuse
    joblib.dump(model, MODEL_PATH) 
    joblib.dump(vect, VECT_PATH)   
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Vectorizer saved to {VECT_PATH}")


if __name__ == "__main__":
    main()
