"""
engineer_team_features.py

Builds a new team_game_stats table in data/nba_data.db, one row per
team per game, with:
  - Rolling averages (5-game, 10-game): points scored, points allowed,
    win rate -- computed from PRIOR games only (no leakage)
  - Home/away career-to-date scoring splits, prior games only
  - rest_days and is_back_to_back (team-level, same logic used in
    engineer_features.py)

Run: python data/engineer_team_features.py
"""

import sqlite3
import pandas as pd

DB_PATH = 'data/nba_data.db'


def load_games(conn):
    return pd.read_sql('SELECT * FROM games', conn)


def reshape_to_team_games(games):
    """One row per team per game, with that team's own score/opponent score."""
    home = games.rename(columns={
        'home_team': 'team',
        'away_team': 'opponent',
        'home_score': 'points_scored',
        'away_score': 'points_allowed',
    })[['game_date', 'team', 'opponent', 'points_scored', 'points_allowed']]
    home['location'] = 'HOME'

    away = games.rename(columns={
        'away_team': 'team',
        'home_team': 'opponent',
        'away_score': 'points_scored',
        'home_score': 'points_allowed',
    })[['game_date', 'team', 'opponent', 'points_scored', 'points_allowed']]
    away['location'] = 'AWAY'

    team_games = pd.concat([home, away], ignore_index=True)
    team_games['won'] = (team_games['points_scored'] > team_games['points_allowed']).astype(int)
    return team_games


def add_rolling_stats(df):
    df = df.sort_values(['team', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('team')

    for window in (5, 10):
        for col, out_name in [
            ('points_scored', f'rolling_pts_scored_{window}'),
            ('points_allowed', f'rolling_pts_allowed_{window}'),
            ('won', f'rolling_win_pct_{window}'),
        ]:
            df[out_name] = grouped[col].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )
    return df


def add_home_away_splits(df):
    df = df.sort_values(['team', 'location', 'game_date']).reset_index(drop=True)
    grouped = df.groupby(['team', 'location'])

    for col, out_name in [
        ('points_scored', 'home_away_pts_scored_avg'),
        ('points_allowed', 'home_away_pts_allowed_avg'),
    ]:
        df[out_name] = grouped[col].transform(lambda x: x.shift(1).expanding().mean())
    return df


def add_rest_and_b2b(df):
    df = df.sort_values(['team', 'game_date']).reset_index(drop=True)
    df['game_date_dt'] = pd.to_datetime(df['game_date'])

    grouped = df.groupby('team')['game_date_dt']
    prev_date = grouped.shift(1)
    df['rest_days'] = (df['game_date_dt'] - prev_date).dt.days
    df['is_back_to_back'] = (df['rest_days'] == 1).astype(int)

    df = df.drop(columns=['game_date_dt'])
    return df


def main():
    conn = sqlite3.connect(DB_PATH)
    games = load_games(conn)

    team_games = reshape_to_team_games(games)
    team_games = add_rolling_stats(team_games)
    team_games = add_home_away_splits(team_games)
    team_games = add_rest_and_b2b(team_games)

    team_games.to_sql('team_game_stats', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    print(f"Built team_game_stats with {len(team_games)} rows (2 rows per game x {len(games)} games)")
    print("Columns:", list(team_games.columns))


if __name__ == '__main__':
    main()
