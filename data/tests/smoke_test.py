from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# Find a player by name
player_list = players.find_players_by_full_name("LeBron James")
print("Found player:", player_list)

lebron_id = player_list[0]['id']

# Pull his game log for the 2024-25 season
gamelog = playergamelog.PlayerGameLog(player_id=lebron_id, season='2024-25', timeout=60)
df = gamelog.get_data_frames()[0]

print(f"\nPulled {len(df)} games")
print(df.head())
