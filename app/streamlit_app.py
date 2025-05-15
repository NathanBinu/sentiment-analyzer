# app/streamlit_app.py

import os
import sys

# ────────────────────────────────────────────────────
# 1) Compute your project root (one level up from app/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2) Add it to sys.path so we can import scripts/*
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import joblib

from scripts.fetch_reddit import fetch_reddit_posts
from scripts.preprocess_and_train import clean_text

# ────────────────────────────────────────────────────
# 3) Paths
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_CSV       = os.path.join(DATA_DIR, "raw_posts.csv")
# ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reddit Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────
st.sidebar.title("Fetch Settings")
sub = st.sidebar.text_input("Subreddit", value="worldnews")
n   = st.sidebar.slider("Posts to fetch", min_value=100, max_value=1000, value=500, step=100)

st.sidebar.markdown("---")
if st.sidebar.button("🔍 Analyze"):
    st.session_state.run = True

# ────────────────────────────────────────────────────
# Load model once
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

pipeline = load_pipeline(PIPELINE_PATH)

# ────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────
# Centered title
st.markdown(
    "<h1 style='text-align: center;'> 🤖 Reddit Sentiment Analyzer 🤖 </h1>",
    unsafe_allow_html=True
)

if not st.session_state.get("run", False):
    st.info("Choose a subreddit and number of posts, then click **Analyze**")
    st.stop()

# step 1: fetch
with st.spinner(f"Fetching {n} posts from r/{sub}…"):
    fetch_reddit_posts(sub, limit=n, out_csv=RAW_CSV)

# step 2: load & clean
df = pd.read_csv(RAW_CSV)
df["cleaned"] = df["text"].fillna("").apply(clean_text)

# step 3: predict
df["label"] = pipeline.predict(df["cleaned"])
df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive", 2: "Neutral"})

# step 4: compute aggregates
counts = df["sentiment"].value_counts().reindex(
    ["Negative", "Neutral", "Positive"], fill_value=0
)
total = counts.sum()
neg_pct = (counts["Negative"] / total) * 100
neu_pct = (counts["Neutral"] / total) * 100
pos_pct = (counts["Positive"] / total) * 100

# ────────────────────────────────────────────────────
# Show summary metrics
col1, col2, col3, col4 = st.columns([1,1,1,2])
col1.metric("Total Posts", f"{total}")
col2.metric("Positive %", f"{pos_pct:.1f}%")
col3.metric("Neutral %", f"{neu_pct:.1f}%")
col4.metric("Negative %", f"{neg_pct:.1f}%")

# ────────────────────────────────────────────────────
# Distribution bar chart
st.subheader("Overall Sentiment Distribution")
st.bar_chart(counts)

# ────────────────────────────────────────────────────
# Trend line chart
st.subheader("Sentiment Over Time")
df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date
trend = df.groupby("date")["sentiment"].value_counts().unstack(fill_value=0)
st.line_chart(trend)

# ────────────────────────────────────────────────────
# Raw data table (collapsible)
with st.expander("🔽 Show Sample Posts", expanded=False):
    st.dataframe(
        df[["date", "text", "sentiment"]]
        .rename(columns={"text": "Title / Selftext"}),
        height=400
    )

# ────────────────────────────────────────────────────
# Updated footer
st.markdown(
    "<div style='text-align: center; margin-top: 2rem;'>"
    "Built by Nathan Binu Edappilly using Streamlit"
    "</div>",
    unsafe_allow_html=True
)
