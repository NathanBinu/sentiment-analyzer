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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  

# ────────────────────────────────────────────────────────────────
# 1) Locate project root so we can import clean_text & RAW_CSV
#    This makes 'scripts' a discoverable package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our text-cleaning function and the path to the raw CSV
from scripts.preprocess_and_train import clean_text, RAW_CSV  
# ─────────────────────────────────────────────────────────────

def load_and_label_df():
    """
    Load the raw CSV of Reddit posts, clean the text,
    and assign a three-way sentiment label using VADER.
    """
    # Read in our combined posts CSV
    df = pd.read_csv(RAW_CSV)
    if df.empty:
        raise RuntimeError(f"No data in {RAW_CSV}; run merge_data first.")

    # # Applied our simple cleanup (removing URLs, mentions, punctuation, lowercase)
    df["cleaned"] = df["text"].fillna("").apply(clean_text)

    # Download VADER lexicon (only the first time)
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    # Compute a 'compound' sentiment score in [-1, 1] for each post
    df["compound"] = df["cleaned"].apply(lambda t: sia.polarity_scores(t)["compound"])
    # Bucket the compound score into labels: Positive (1), Negative (0), Neutral (2)
    df["label"] = df["compound"].apply(
        lambda c: 1 if c >=  0.05 else (0 if c <= -0.05 else 2)
    )

    # Return only the cleaned text and its assigned label
    return df[["cleaned", "label"]]

# ────────────────────────────────────────────────────────────────
# new: compute_metrics callback for Trainer
def compute_metrics(eval_pred):
    """
    Callback for the Trainer to compute accuracy, precision, recall, and F1.
    """
    logits, labels = eval_pred
    # Converting raw model outputs to predicted class indices
    preds = logits.argmax(axis=-1)
    # Computing standard classification metrics
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
    # 1) Prepare DataFrame with cleaned text + VADER labels
    df = load_and_label_df()

    # 2) Split into 80% train / 20% test, preserving label distribution
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3) Convert pandas DataFrames into Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # 4) Load the pretrained DistilBERT model and tokenizer for 3-way classification
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # 5) Tokenization function: convert text to token IDs
    def tokenize(batch):
        return tokenizer(batch["cleaned"], truncation=True)

    # Apply tokenization in batch for speed    
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    # 6) Use a data collator to dynamically pad batches to the correct length
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7) Set up training arguments (where to save, batch sizes, epochs, logging)
    output_dir = os.path.join(PROJECT_ROOT, "models", "reddit-bert")
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_steps=100,      # log metrics every 100 steps
        do_train=True,
        do_eval=True,
        eval_steps=500,         # run evaluation every 500 steps
        save_steps=500,         # checkpoint every 500 steps
        save_total_limit=1,     # only keep the most recent checkpoint
        push_to_hub=False,      # disable automatic Hugging Face hub upload
    )

    # 8) Create the Trainer: ties together model, data, args, and metrics
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,    # custom metric computation
    )

    # 9) Kicking off the fine-tuning!
    trainer.train()

    # 10) Evaluate the final model on the test set
    metrics = trainer.evaluate()
    print("✅ Final evaluation metrics:", metrics)

    # 11) Save the fine-tuned model to disk for later use in the app
    trainer.save_model(output_dir)
    print(f"✅ Fine-tuned Reddit-BERT saved to {output_dir}")

if __name__ == "__main__":
    main()
