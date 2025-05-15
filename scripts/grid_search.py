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
# 1) Locating project root & make sure we can import your preprocess code
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the clean_text function and the path to our raw CSV
from scripts.preprocess_and_train import clean_text, RAW_CSV  # noqa: E402

# 2) Set up VADER for auto-labeling
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# ────────────────────────────────────────────────────────────────

def load_and_label():
    """
    1) Read in the raw combined CSV of Reddit posts
    2) Clean each post’s text
    3) Auto-label with VADER’s compound score (Positive/Neutral/Negative)
    Returns the full DataFrame with a new `cleaned` and `label` column.
    """
    # Load the data
    df = pd.read_csv(RAW_CSV)

    # Apply our text-cleaning function
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # Compute VADER compound scores and bucket them into labels:
    # 1 if compound >= 0.05, 0 if compound <= -0.05, else 2
    df["compound"] = df["text"].fillna("").apply(lambda t: sia.polarity_scores(t)["compound"])
    df["label"] = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    return df

def main():
    # Load and label our data
    df = load_and_label()
    X = df["cleaned"]  # the cleaned text we’ll feed into the model
    y = df["label"]  # our target labels

    # 3) Building a pipeline: TF–IDF vectorizer + Logistic Regression classifier
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),            # turn text into numerical features
        ("clf",  LogisticRegression(max_iter=500,            # Simple linear classifier
                                    class_weight="balanced")) # auto-adjust for class imbalance
        
    ])

    # The “grid” of hyperparameters we want to try:
    # - ngram range: unigrams, bigrams, trigrams
    # - max_df/min_df to filter out overly common or rare words
    # - C: regularization strength for the classifier
    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
        "tfidf__max_df": [0.7, 0.8, 0.9],
        "tfidf__min_df": [3, 5, 10],
        "clf__C": [0.1, 1, 10],
    }

    # 4) GridSearchCV runs a 5-fold cross-validation over every combination
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,                 # 5 splits of the data
        scoring="f1_weighted",      # optimize for weighted F1 score
        n_jobs=-1,          # use all CPU cores
        verbose=1           # show progress messages
    )
    grid.fit(X, y)        # this is where all the training & validation happens

    # Report the best hyperparameters and their cross-validated F1 score
    print("\n Best parameters:", grid.best_params_)
    print("Best CV f1_weighted score:", grid.best_score_)

    # 5) Save the winning Pipeline (vectorizer + classifier) to disk
    out_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    joblib.dump(grid.best_estimator_, out_path)
    print(f"Tuned model saved to {out_path}")

    # 6) Honest test‐set evaluation on a held-out 20% split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = grid.best_estimator_.predict(X_test)
    print("\n Test‐set performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
