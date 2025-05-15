

import pandas as pd
import os

# List the files you just created
files = [
    "data/raw_worldnews.csv",
    "data/raw_technology.csv",
    "data/raw_funny.csv"
]

dfs = []
for f in files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f} – did you run fetch_reddit with --out {f}?")
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
out = "data/raw_combined.csv"
df.to_csv(out, index=False)
print(f"✅ Combined {len(files)} files → {df.shape[0]} rows in {out}")
