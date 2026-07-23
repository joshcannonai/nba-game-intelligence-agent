"""
load_to_db.py

Builds data/nba_data.db from raw CSVs produced by pull_games.py and
pull_player_stats_range.py. Creates schema for games, player_stats,
and injuries (injuries loader deferred until a historical source is found).

Run: python data/load_to_db.py
"""

import glob
import sqlite3
import pandas as pd

DB_PATH = 'data/nba_data.db'
GAMES_CSV = 'data/raw/season_schedule_2026.csv'
PLAYER_STATS_GLOB = 'data/raw/player_box_scores_*.csv'


def create_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time_utc TEXT,
        game_date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_score INTEGER,
        away_score INTEGER
    );

    CREATE TABLE IF NOT EXISTS player_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slug TEXT,
        name TEXT,
        team TEXT,
        opponent TEXT,
        location TEXT,
        outcome TEXT,
        game_date TEXT,
        seconds_played INTEGER,
        made_field_goals INTEGER,
        attempted_field_goals INTEGER,
        made_three_point_field_goals INTEGER,
        attempted_three_point_field_goals INTEGER,
        made_free_throws INTEGER,
        attempted_free_throws INTEGER,
        offensive_rebounds INTEGER,
        defensive_rebounds INTEGER,
        assists INTEGER,
        steals INTEGER,
        blocks INTEGER,
        turnovers INTEGER,
        personal_fouls INTEGER,
        plus_minus REAL,
        game_score REAL
    );

    CREATE TABLE IF NOT EXISTS injuries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT,
        team TEXT,
        status TEXT,
        injury_type TEXT,
        return_date TEXT,
        date_reported TEXT,
        comment TEXT
    );
    """)
    conn.commit()


def clean_team_name(series):
    """Strip 'Team.' prefix, e.g. 'Team.BOSTON_CELTICS' -> 'BOSTON_CELTICS'."""
    return series.astype(str).str.replace('Team.', '', regex=False)


def load_games(conn, csv_path):
    df = pd.read_csv(csv_path)

    # Parse UTC timestamps (strings already carry +00:00 offset)
    df['start_time'] = pd.to_datetime(df['start_time'])

    # Convert to US/Eastern and take the calendar date -> matches
    # basketball_reference_web_scraper's local game_date convention
    df['game_date'] = df['start_time'].dt.tz_convert('US/Eastern').dt.date.astype(str)

    df['home_team'] = clean_team_name(df['home_team'])
    df['away_team'] = clean_team_name(df['away_team'])

    out = pd.DataFrame({
        'start_time_utc': df['start_time'].astype(str),
        'game_date': df['game_date'],
        'home_team': df['home_team'],
        'away_team': df['away_team'],
        'home_score': df['home_team_score'],
        'away_score': df['away_team_score'],
    })

    out.to_sql('games', conn, if_exists='append', index=False)
    print(f"Loaded {len(out)} rows into games")


def load_player_stats(conn, csv_glob):
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {csv_glob}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Combined {len(files)} daily files into {len(df)} rows")

    df['team'] = clean_team_name(df['team'])
    df['opponent'] = clean_team_name(df['opponent'])
    df['location'] = df['location'].astype(str).str.replace('Location.', '', regex=False)
    df['outcome'] = df['outcome'].astype(str).str.replace('Outcome.', '', regex=False)

    out = df.rename(columns={'game_date': 'game_date'})[[
        'slug', 'name', 'team', 'opponent', 'location', 'outcome', 'game_date',
        'seconds_played', 'made_field_goals', 'attempted_field_goals',
        'made_three_point_field_goals', 'attempted_three_point_field_goals',
        'made_free_throws', 'attempted_free_throws', 'offensive_rebounds',
        'defensive_rebounds', 'assists', 'steals', 'blocks', 'turnovers',
        'personal_fouls', 'plus_minus', 'game_score'
    ]]

    out.to_sql('player_stats', conn, if_exists='append', index=False)
    print(f"Loaded {len(out)} rows into player_stats")


def validate_join(conn):
    """Sanity check: how many player_stats rows fail to find a matching game?"""
    query = """
    SELECT COUNT(*) FROM player_stats ps
    WHERE NOT EXISTS (
        SELECT 1 FROM games g
        WHERE g.game_date = ps.game_date
        AND (
            (g.home_team = ps.team AND g.away_team = ps.opponent)
            OR (g.away_team = ps.team AND g.home_team = ps.opponent)
        )
    )
    """
    unmatched = conn.execute(query).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
    print(f"Join check: {unmatched} of {total} player_stats rows have no matching game")


def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    load_games(conn, GAMES_CSV)
    load_player_stats(conn, PLAYER_STATS_GLOB)
    validate_join(conn)
    conn.close()


if __name__ == '__main__':
    main()
