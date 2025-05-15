import os
import sys
import pandas as pd
import nltk
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # ← new

# ────────────────────────────────────────────────────────────────
# 1) Locate project root so we can import clean_text & RAW_CSV
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess_and_train import clean_text, RAW_CSV  # noqa: E402
# ────────────────────────────────────────────────────────────────

def load_and_label_df():
    # Load the merged CSV
    df = pd.read_csv(RAW_CSV)
    if df.empty:
        raise RuntimeError(f"No data in {RAW_CSV}; run merge_data first.")

    # Clean text
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # Label via VADER
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    df["compound"] = df["cleaned"].apply(lambda t: sia.polarity_scores(t)["compound"])
    df["label"]    = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    return df[["cleaned", "label"]]

# ────────────────────────────────────────────────────────────────
# new: compute_metrics callback for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy":    acc,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
    }
# ────────────────────────────────────────────────────────────────

def main():
    # 1) Prepare DataFrame
    df = load_and_label_df()

    # 2) Train/test split (80/20 stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3) HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # 4) Tokenizer & Model
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # 5) Tokenization step
    def tokenize(batch):
        return tokenizer(batch["cleaned"], truncation=True)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    # 6) Dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7) TrainingArguments
    output_dir = os.path.join(PROJECT_ROOT, "models", "reddit-bert")
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_steps=100,
        do_train=True,
        do_eval=True,
        eval_steps=500,         # run evaluation every 500 steps
        save_steps=500,         # checkpoint every 500 steps
        save_total_limit=1,
        push_to_hub=False,
    )

    # 8) Trainer (now with compute_metrics)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,    # ← added
    )

    # 9) Train!
    trainer.train()

    # 10) Evaluate final model
    metrics = trainer.evaluate()
    print("✅ Final evaluation metrics:", metrics)

    # 11) Save final model
    trainer.save_model(output_dir)
    print(f"✅ Fine-tuned Reddit-BERT saved to {output_dir}")

if __name__ == "__main__":
    main()
