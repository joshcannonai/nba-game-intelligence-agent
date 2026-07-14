from basketball_reference_web_scraper import client
import pandas as pd

games = client.season_schedule(season_end_year=2026)
df = pd.DataFrame(games)
df.to_csv('data/raw/season_schedule_2026.csv', index=False)
print(f"Pulled {len(df)} games")
