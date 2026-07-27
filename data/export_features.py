"""
export_features.py

Exports player_stats and team_game_stats from data/nba_data.db to CSV
files for sharing with teammates who want flat files instead of
querying the database directly.

Run: python data/export_features.py
"""

import os
import sqlite3
import pandas as pd

DB_PATH = 'data/nba_data.db'
EXPORT_DIR = 'data/exports'


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    player_df = pd.read_sql('SELECT * FROM player_stats', conn)
    team_df = pd.read_sql('SELECT * FROM team_game_stats', conn)

    conn.close()

    player_path = os.path.join(EXPORT_DIR, 'player_stats_engineered.csv')
    team_path = os.path.join(EXPORT_DIR, 'team_game_stats_engineered.csv')

    player_df.to_csv(player_path, index=False)
    team_df.to_csv(team_path, index=False)

    print(f"Exported {len(player_df)} rows to {player_path}")
    print(f"Exported {len(team_df)} rows to {team_path}")


if __name__ == '__main__':
    main()
