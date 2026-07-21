"""
engineer_features.py

Adds feature columns directly onto player_stats in data/nba_data.db:
  - Rolling averages (5-game, 10-game): points, rebounds, assists,
    minutes, FG%, 3P% -- computed from PRIOR games only (no leakage)
  - Home/away career-to-date splits, prior games only
  - Team-level back-to-back flag and rest-days count

Run: python data/engineer_features.py
"""

import sqlite3
import pandas as pd

DB_PATH = 'data/nba_data.db'


def load_tables(conn):
    player_stats = pd.read_sql('SELECT * FROM player_stats', conn)
    games = pd.read_sql('SELECT * FROM games', conn)
    return player_stats, games


def add_derived_totals(df):
    """Points and total rebounds aren't in the raw data -- derive them."""
    df['points'] = (
        2 * df['made_field_goals']
        + df['made_three_point_field_goals']
        + df['made_free_throws']
    )
    df['total_rebounds'] = df['offensive_rebounds'] + df['defensive_rebounds']
    df['minutes'] = df['seconds_played'] / 60.0
    return df


def add_rolling_averages(df):
    """Rolling mean/pct over a player's last N games, excluding the current game."""
    df = df.sort_values(['slug', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('slug')

    for window in (5, 10):
        for col, out_name in [
            ('points', f'rolling_pts_{window}'),
            ('total_rebounds', f'rolling_reb_{window}'),
            ('assists', f'rolling_ast_{window}'),
            ('minutes', f'rolling_min_{window}'),
        ]:
            df[out_name] = grouped[col].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )

        # Shooting percentages: sum makes/attempts over the window, then divide once
        # (averaging per-game percentages would misweight low-attempt games)
        made_fg = grouped['made_field_goals'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        att_fg = grouped['attempted_field_goals'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        df[f'rolling_fg_pct_{window}'] = (made_fg / att_fg.replace(0, pd.NA))

        made_3p = grouped['made_three_point_field_goals'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        att_3p = grouped['attempted_three_point_field_goals'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        df[f'rolling_3p_pct_{window}'] = (made_3p / att_3p.replace(0, pd.NA))

    return df


def add_home_away_splits(df):
    """Career-to-date average by location (HOME/AWAY), prior games only."""
    df = df.sort_values(['slug', 'location', 'game_date']).reset_index(drop=True)
    grouped = df.groupby(['slug', 'location'])

    for col, out_name in [
        ('points', 'home_away_pts_avg'),
        ('total_rebounds', 'home_away_reb_avg'),
        ('assists', 'home_away_ast_avg'),
    ]:
        df[out_name] = grouped[col].transform(lambda x: x.shift(1).expanding().mean())

    return df


def build_team_schedule(games):
    """One row per team per game_date, with days since that team's previous game."""
    home = games[['game_date', 'home_team']].rename(columns={'home_team': 'team'})
    away = games[['game_date', 'away_team']].rename(columns={'away_team': 'team'})
    team_games = pd.concat([home, away], ignore_index=True).drop_duplicates()

    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    team_games = team_games.sort_values(['team', 'game_date'])

    team_games['prev_game_date'] = team_games.groupby('team')['game_date'].shift(1)
    team_games['rest_days'] = (team_games['game_date'] - team_games['prev_game_date']).dt.days
    team_games['is_back_to_back'] = (team_games['rest_days'] == 1).astype(int)

    # First game of the season for a team has no prior game -- leave rest_days null,
    # not back-to-back
    team_games['game_date'] = team_games['game_date'].dt.strftime('%Y-%m-%d')
    return team_games[['team', 'game_date', 'rest_days', 'is_back_to_back']]


def add_team_context(df, team_schedule):
    df = df.merge(
        team_schedule,
        left_on=['team', 'game_date'],
        right_on=['team', 'game_date'],
        how='left'
    )
    return df


def main():
    conn = sqlite3.connect(DB_PATH)
    player_stats, games = load_tables(conn)

    player_stats = add_derived_totals(player_stats)
    player_stats = add_rolling_averages(player_stats)
    player_stats = add_home_away_splits(player_stats)

    team_schedule = build_team_schedule(games)
    player_stats = add_team_context(player_stats, team_schedule)

    player_stats.to_sql('player_stats', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    print(f"Updated player_stats with {len(player_stats)} rows and new feature columns")
    print("New columns:", [c for c in player_stats.columns if c.startswith(('rolling_', 'home_away_', 'rest_', 'is_back'))])


if __name__ == '__main__':
    main()
