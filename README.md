#  Reddit Sentiment Analyzer

An end-to-end pipeline that fetches live Reddit posts, auto-labels them with VADER, trains two sentiment models (TF–IDF + Logistic Regression and an optional DistilBERT fine-tune), and serves results in an interactive Streamlit dashboard.

---

## 1. Introduction

With the explosion of user-generated content on Reddit, automatic sentiment analysis can help you understand community mood, track brand perceptions, or monitor public reaction to world events. This project demonstrates:

- **Data collection** from any subreddit  
- **Silver labeling** via NLTK’s rule-based VADER  
- **Baseline training** using TF–IDF + Logistic Regression  
- **Hyperparameter tuning** with scikit-learn’s GridSearchCV  
- **(Optional)** Context-aware Transformer fine-tuning (DistilBERT)  
- **Deployment** in a Streamlit dashboard  

---

## 2. Models Used

1. **Baseline (TF–IDF + Logistic Regression)**  
   - Converts text to TF–IDF vectors (bag-of-words)  
   - Trains a linear classifier on VADER-generated labels  
   - Fast to train, straightforward to interpret  

2. **Transformer (DistilBERT fine-tune)**   
   - Starts from `distilbert-base-uncased` pre-trained weights  
   - Fine-tunes on your Reddit data for three-way sentiment  
   - Captures linguistic nuance and context for higher accuracy  

---

## 3. Features

- **Live data fetch** from any subreddit via the Reddit API (PRAW)  
- **Automatic “silver” labeling** using VADER’s compound polarity scores  
- **Class balancing** by up-sampling minority sentiment classes  
- **Baseline model** artifacts: `models/baseline.pkl` + `models/vectorizer.pkl`  
- **Tuned pipeline**: best TF–IDF + LR parameters → `models/best_model.pkl`  
- **Optional Transformer** directory: `models/reddit-bert/`  
- **Streamlit dashboard** with bar charts, line charts, and sample post table  

---

## 4. Prerequisites

- **Python 3.8+**  
- **Reddit API credentials** (client ID, client secret, user agent)  
- **(Optional)** GPU & CUDA for faster Transformer fine-tuning  
- **Network access** to fetch live Reddit data  

---

## 5. Getting the Code

```bash
# SSH clone (we’ve already set up your SSH key)
git clone git@github.com:NathanBinu/reddit-sentiment-analyzer.git
cd reddit-sentiment-analyzer

# Or HTTPS clone if you prefer:
git clone https://github.com/NathanBinu/reddit-sentiment-analyzer.git
```


**We store the 3 big model files (baseline.pkl, vectorizer.pkl, best_model.pkl) in Git LFS. After you clone, please run:**
```bash
git lfs install        # only needs to be run once on your machine
git lfs pull           # downloads the actual .pkl/.safetensors binaries
```
## 5. Installation & Setup

1. Clone repo  

```bash
git clone https://github.com/NathanBinu/reddit-sentiment-analyzer.git
cd reddit-sentiment-analyzer
```
2. Create & activate virtualenv  
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies  
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. Configure Reddit API credentials  

```bash
# config.py
REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REDDIT_USER_AGENT    = "sentiment-analyzer-script by /u/YourRedditUsername"
```

## 6. Usage

### 6.1 Fetch Posts  

```bash
python -m scripts/fetch_reddit \
  --sub worldnews \
  --limit 1000 \
  --out data/raw_worldnews.csv
```
    Terms:
    --sub: Subreddit name
    --limit: Max posts to fetch
    --out: Where to save the CSV

### 6.2 Merge Data  

```bash
python -m scripts/merge_data \
  --files data/raw_worldnews.csv data/raw_technology.csv \
  --out data/raw_combined.csv
```
    Terms:
    --files: space-seperated list of input CSVs
    --out: combined output CSV

### 6.3 Choose & Train Model  

#### Option A: 
```bash
python -m scripts/preprocess_and_train
# (Optional) Hyperparameter Tuning:
python -m scripts/grid_search
```

    Outputs:
    models/baseline.pkl [Trained Classifier]
    models/vectorizer.pkl [TF-IDF Transformer]
    models/best_model.pkl [Optional Hyperparameter Tuning]

#### Option B:
```bash
python -m scripts/finetune_reddit
```
    Outputs:
    models/reddit-bert [fine-tuned weights & config]

### 6.4 Hyperparameter Tuning  

```bash
python -m scripts/grid_search
```
    Outputs:
    models/best_model.pkl [Tuned Pipeline]
    Console logs showing best parameters, CV F₁ score, and test-set performance

### 6.5 Fine-Tune Transformer   
```bash
python -m scripts/finetune_reddit
```
    Outputs:
    Directory models/reddit-bert/ containing the fine-tuned model

### 6.6 Launch Dashboard  
```bash
streamlit run app/streamlit_app.py
```

## 7. Steps to run:

### Step 1: Enter your target Subreddit and Number of posts in the sidebar.
### Step 2: Click Analyze.
### Step 3: View:
- Metrics: total posts & % Positive / Neutral / Negative
- Bar Chart: Overall sentiment distribution
- Line Chart: Sentiment trend over time 
- Sample Posts: Table with predicted labels