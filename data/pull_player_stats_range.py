from basketball_reference_web_scraper import client
import pandas as pd
from datetime import date, timedelta
import argparse
import time
import os

# Defaults are the 2025-26 pull this script originally did. The date range is an
# argument now because predict_stat_line needs seasons the agent is NOT replaying:
# a stat-line model fitted on 2025-26 and then used to predict 2025-26 games would
# leak, the same way a current-season team rating leaks (see agent/sources.py).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", default="2025-10-21")
parser.add_argument("--end", default="2026-04-12")
parser.add_argument("--out-dir", default="data/raw")
args = parser.parse_args()

start = date.fromisoformat(args.start)
end = date.fromisoformat(args.end)
os.makedirs(args.out_dir, exist_ok=True)

current = start
while current <= end:
    filename = f"{args.out_dir}/player_box_scores_{current.isoformat()}.csv"

    if os.path.exists(filename):
        print(f"{current}: already pulled, skipping")
        current += timedelta(days=1)
        continue

    try:
        box_scores = client.player_box_scores(
            day=current.day, month=current.month, year=current.year
        )
        if box_scores:
            df = pd.DataFrame(box_scores)
            df["game_date"] = current.isoformat()
            df.to_csv(filename, index=False)
            print(f"{current}: {len(df)} performances")
        else:
            print(f"{current}: no games")
    except Exception as e:
        print(f"{current}: FAILED - {e}")
        print("Waiting 60s before retrying...")
        time.sleep(60)
        continue  # retries same date without advancing

    time.sleep(3)  # be polite between requests
    current += timedelta(days=1)
