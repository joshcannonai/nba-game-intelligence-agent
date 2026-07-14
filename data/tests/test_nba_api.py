from nba_api.stats.endpoints import commonteamroster
import time

start = time.time()
try:
    roster = commonteamroster.CommonTeamRoster(team_id=1610612747, timeout=10)  # Lakers
    df = roster.get_data_frames()[0]
    print(f"SUCCESS in {time.time()-start:.1f}s — {len(df)} rows")
    print(df.head())
except Exception as e:
    print(f"FAILED after {time.time()-start:.1f}s — {type(e).__name__}: {e}")
