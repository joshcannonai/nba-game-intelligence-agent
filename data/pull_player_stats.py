from basketball_reference_web_scraper import client
import pandas as pd
from datetime import date

# Pull box scores for a single day first to test
box_scores = client.player_box_scores(day=21, month=10, year=2025)
df = pd.DataFrame(box_scores)
print(f"Pulled {len(df)} player performances")
print(df.head())
