from basketball_reference_web_scraper import client
import pandas as pd
from datetime import date, timedelta
import time
import os

start = date(2025, 10, 21)
end = date(2026, 4, 12)

current = start
while current <= end:
    filename = f'data/raw/player_box_scores_{current.isoformat()}.csv'

    if os.path.exists(filename):
        print(f"{current}: already pulled, skipping")
        current += timedelta(days=1)
        continue

    try:
        box_scores = client.player_box_scores(day=current.day, month=current.month, year=current.year)
        if box_scores:
            df = pd.DataFrame(box_scores)
            df['game_date'] = current.isoformat()
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
