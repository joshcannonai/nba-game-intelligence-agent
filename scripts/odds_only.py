import pandas as pd
from pathlib import Path

# Where to read from, and where to save the clean file
RAW_FILE = "data/raw/odds/primary/nba_2008-2026.csv"
OUTPUT_FILE = "data/samples/odds_only.csv"

# Only these columns are safe (no scores)
SAFE_COLUMNS = [
    "season", "date", "regular", "playoffs", "away", "home",
    "whos_favored", "spread", "total",
    "moneyline_away", "moneyline_home",
    "h2_spread", "h2_total", "id_spread", "id_total",
]

# Read the raw file
df = pd.read_csv(RAW_FILE)

# Keep only the safe columns
safe_df = df[SAFE_COLUMNS]

# Save the clean version
Path("data/samples").mkdir(parents=True, exist_ok=True)
safe_df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Saved {len(safe_df)} rows to {OUTPUT_FILE}")
