from basketball_reference_web_scraper import client
import time

start = time.time()
try:
    games = client.season_schedule(season_end_year=2026)
    print(f"SUCCESS in {time.time()-start:.1f}s — {len(games)} games")
    print(games[0])
except Exception as e:
    print(f"FAILED after {time.time()-start:.1f}s — {type(e).__name__}: {e}")
