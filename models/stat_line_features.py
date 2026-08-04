"""Player-level features for the stat-line model, built the way Patrick built them.

The rolling/split logic is imported from `data/engineer_features.py` rather than
rewritten here. That file is the data layer's definition of "a player's form as of
this game", and a second implementation would drift from it silently -- the model
would then be trained on features that no longer match what the tool serves.

WHAT MAKES THESE AS-OF. Every feature is a `shift(1)` over the player's own prior
games, so the row for a game on 2024-03-01 contains only what was knowable on the
morning of 2024-03-01. The target (points, rebounds, assists) is that game's actual
line. Training pairs the two; inference gets the features alone.

WHY THE TARGET IS STRIPPED AT INFERENCE. `data/exports/player_stats_engineered.csv`
keeps the features and the box-score result in the same row, exactly like the odds
file keeps `score_home` next to the line. That is the single most likely way this
tool could leak, so `build_features_only()` writes a file with the answer columns
removed and `tests/test_stat_line.py` asserts they are gone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))

from engineer_features import (  # noqa: E402
    add_derived_totals,
    add_home_away_splits,
    add_rolling_averages,
)

FEATURE_NAMES = (
    "rolling_pts_5",
    "rolling_reb_5",
    "rolling_ast_5",
    "rolling_min_5",
    "rolling_pts_10",
    "rolling_reb_10",
    "rolling_ast_10",
    "rolling_min_10",
    "rolling_fg_pct_5",
    "rolling_3p_pct_5",
    "home_away_pts_avg",
    "home_away_reb_avg",
    "home_away_ast_avg",
    "rest_days",
    "is_back_to_back",
    "is_home",
)

TARGETS = ("points", "total_rebounds", "assists")

# Columns that reveal the outcome of the game being predicted. Anything here is
# dropped before a features file is written. Keep this list ahead of the schema:
# a new box-score column is guilty until someone checks it.
LEAK_COLUMNS = (
    "points",
    "total_rebounds",
    "assists",
    "outcome",
    "plus_minus",
    "game_score",
    "seconds_played",
    "minutes",
    "made_field_goals",
    "attempted_field_goals",
    "made_three_point_field_goals",
    "attempted_three_point_field_goals",
    "made_free_throws",
    "attempted_free_throws",
    "offensive_rebounds",
    "defensive_rebounds",
    "steals",
    "blocks",
    "turnovers",
    "personal_fouls",
)

IDENTITY_COLUMNS = ("slug", "name", "team", "opponent", "location", "game_date")


def _strip_enum(value):
    """`Team.DENVER_NUGGETS` -> `DENVER_NUGGETS`. The scraper emits Python enums."""
    if isinstance(value, str) and "." in value:
        head, _, tail = value.partition(".")
        if head in {"Team", "Location", "Outcome"}:
            return tail
    return value


def load_box_scores(directory: Path) -> pd.DataFrame:
    """Every `player_box_scores_*.csv` in one directory, concatenated."""
    paths = sorted(directory.glob("player_box_scores_*.csv"))
    if not paths:
        raise SystemExit(
            f"no player_box_scores_*.csv in {directory}. Pull them first:\n"
            "  python data/pull_player_stats_range.py --start ... --end ... "
            "--out-dir ..."
        )
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    for col in ("team", "opponent", "location", "outcome"):
        if col in df.columns:
            df[col] = df[col].map(_strip_enum)
    return df


def add_team_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game, derived from the box scores themselves.

    `engineer_features.build_team_schedule` wants a `games` table we do not have
    here, but every team-date pair is already present in the box scores, so the
    schedule is recoverable from them.
    """
    team_games = df[["team", "game_date"]].drop_duplicates().copy()
    team_games["game_date"] = pd.to_datetime(team_games["game_date"])
    team_games = team_games.sort_values(["team", "game_date"])
    prev = team_games.groupby("team")["game_date"].shift(1)
    team_games["rest_days"] = (team_games["game_date"] - prev).dt.days
    team_games["is_back_to_back"] = (team_games["rest_days"] == 1).astype(int)
    team_games["game_date"] = team_games["game_date"].dt.strftime("%Y-%m-%d")
    return df.merge(team_games, on=["team", "game_date"], how="left")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw box scores -> one row per player-game with as-of features and targets."""
    df = add_derived_totals(df)
    df = add_rolling_averages(df)
    df = add_home_away_splits(df)
    df = add_team_rest(df)
    df["is_home"] = (df["location"] == "HOME").astype(int)
    return df.sort_values(["game_date", "slug"]).reset_index(drop=True)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where every feature is known and the player actually took the floor.

    A player's first game of the range has no prior games, so its `shift(1)`
    features are null. Those rows are dropped rather than imputed: a made-up
    prior average is the same class of invented number the tool interface exists
    to prevent.
    """
    df = df[df["seconds_played"] > 0]
    return df.dropna(subset=list(FEATURE_NAMES) + list(TARGETS))


def build_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Identity + features, with every outcome column removed. See module docstring."""
    keep = [c for c in IDENTITY_COLUMNS if c in df.columns] + list(FEATURE_NAMES)
    out = df[keep].copy()
    leaked = [c for c in out.columns if c in LEAK_COLUMNS]
    if leaked:  # belt and braces -- IDENTITY/FEATURE lists should never contain these
        raise AssertionError(f"outcome columns survived the strip: {leaked}")
    return out


# --- inference data -------------------------------------------------------------
#
# The model trains on 2023-24 and 2024-25. It INFERS on 2025-26, which is the season
# the agent replays, so that file has to be safe for the agent to read: features
# only, no box-score result. Patrick's engineered export already carries the rolling
# columns, so this is a strip, not a recomputation.

ENGINEERED_CSV = REPO_ROOT / "data" / "exports" / "player_stats_engineered.csv"
INFERENCE_CSV = REPO_ROOT / "data" / "samples" / "player_features_2026.csv"


def build_inference_file(
    source: Path = ENGINEERED_CSV, dest: Path = INFERENCE_CSV
) -> pd.DataFrame:
    """Write the features-only 2025-26 file the agent's tool reads."""
    df = pd.read_csv(source)
    df["is_home"] = (df["location"].map(_strip_enum) == "HOME").astype(int)
    out = build_features_only(df).dropna(subset=list(FEATURE_NAMES))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out


def main() -> None:
    out = build_inference_file()
    print(f"wrote {INFERENCE_CSV.relative_to(REPO_ROOT)}: {len(out)} player-games")
    print(f"date range {out['game_date'].min()} -> {out['game_date'].max()}")
    print(f"columns ({len(out.columns)}): {list(out.columns)}")


if __name__ == "__main__":
    main()
