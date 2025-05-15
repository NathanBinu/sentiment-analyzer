import os, csv, argparse
import praw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

"""
    Connects to Reddit, fetches up to `limit` unique posts from the given subreddit,
    and writes them to a CSV file at `out_csv`.
"""

def fetch_reddit_posts(subreddit_name, limit=1000, out_csv="data/raw_posts.csv"):

    # Initialize the PRAW Reddit client with your credentials
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    # Referencing the subreddit object
    sub = reddit.subreddit(subreddit_name)

    # ensure output dir exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    seen_ids = set()
    collected = []

    """
        Loops through a list of posts, adds each new one to our collection,
        and stops if we've reached the requested limit.
    """
    def add_posts(posts):
        for post in posts:
            # Skip any post we've already seen
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                # Combine title and body (selftext) into one text field
                text = post.title + " " + (post.selftext or "")
                collected.append((post.id, post.created_utc, text))
            # If we've hit our fetch limit, break out early    
            if len(collected) >= limit:
                break

    # 1) Grab the freshest posts
    add_posts(sub.new(limit=limit))

    # 2) If still short, grab all-time top
    if len(collected) < limit:
        add_posts(sub.top(time_filter="all", limit=limit))

    # 3) Write out up to the limit
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "created_utc", "text"])
        for row in collected[:limit]:
            writer.writerow(row)

    print(f"Fetched {len(collected[:limit])} unique posts to {out_csv}")

if __name__ == "__main__":
    # Simple command-line interface
    p = argparse.ArgumentParser(description="Fetch Reddit posts via PRAW + merge")
    p.add_argument("--sub",   required=True,  help="Subreddit name (no r/)")
    p.add_argument("--limit", type=int, default=1000, help="Total posts to fetch")
    p.add_argument("--out",   default="data/raw_posts.csv", help="Output CSV path (e.g. data/raw_worldnews.csv)")
    args = p.parse_args()

    # Runing the fetch function with the user’s arguments
    fetch_reddit_posts(
        subreddit_name=args.sub,
        limit=args.limit,
        out_csv=args.out
    )
