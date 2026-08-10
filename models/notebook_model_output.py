from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agent.sources import (
    parse_date,
    parse_matchup_id as parse_source_matchup_id,
    player_is_out,
)

ROOT = Path(__file__).resolve().parents[1]
PLAYER_CSV = ROOT / "data" / "exports" / "player_stats_engineered.csv"
TEAM_CSV = ROOT / "data" / "exports" / "team_game_stats_engineered.csv"

TEAM_KEYS = {
    "ATL": "ATLANTA_HAWKS",
    "BOS": "BOSTON_CELTICS",
    "BRK": "BROOKLYN_NETS",
    "BKN": "BROOKLYN_NETS",
    "CHA": "CHARLOTTE_HORNETS",
    "CHO": "CHARLOTTE_HORNETS",
    "CHI": "CHICAGO_BULLS",
    "CLE": "CLEVELAND_CAVALIERS",
    "DAL": "DALLAS_MAVERICKS",
    "DEN": "DENVER_NUGGETS",
    "DET": "DETROIT_PISTONS",
    "GSW": "GOLDEN_STATE_WARRIORS",
    "HOU": "HOUSTON_ROCKETS",
    "IND": "INDIANA_PACERS",
    "LAC": "LA_CLIPPERS",
    "LAL": "LOS_ANGELES_LAKERS",
    "MEM": "MEMPHIS_GRIZZLIES",
    "MIA": "MIAMI_HEAT",
    "MIL": "MILWAUKEE_BUCKS",
    "MIN": "MINNESOTA_TIMBERWOLVES",
    "NOP": "NEW_ORLEANS_PELICANS",
    "NYK": "NEW_YORK_KNICKS",
    "OKC": "OKLAHOMA_CITY_THUNDER",
    "ORL": "ORLANDO_MAGIC",
    "PHI": "PHILADELPHIA_76ERS",
    "PHX": "PHOENIX_SUNS",
    "PHO": "PHOENIX_SUNS",
    "POR": "PORTLAND_TRAIL_BLAZERS",
    "SAC": "SACRAMENTO_KINGS",
    "SAS": "SAN_ANTONIO_SPURS",
    "TOR": "TORONTO_RAPTORS",
    "UTA": "UTAH_JAZZ",
    "WAS": "WASHINGTON_WIZARDS",
}

PLAYER_FEATURE_COLS = [
    "rolling_pts_5",
    "rolling_reb_5",
    "rolling_ast_5",
    "rolling_min_5",
    "rolling_fg_pct_5",
    "rolling_3p_pct_5",
    "rolling_pts_10",
    "rolling_reb_10",
    "rolling_ast_10",
    "rolling_min_10",
    "rolling_fg_pct_10",
    "rolling_3p_pct_10",
    "home_away_pts_avg",
    "home_away_reb_avg",
    "home_away_ast_avg",
    "rest_days",
    "is_back_to_back",
    "is_home",
]

PLAYER_TARGETS = {
    "predicted_points": "points",
    "predicted_rebounds": "total_rebounds",
    "predicted_assists": "assists",
}

# Deliberately explicit: these columns are produced with shift(1) in
# data/engineer_team_features.py, so they are available before tipoff.  Do not
# replace this with dtype-based discovery; the export also contains the target
# game's final score, which must never become a model input.
TEAM_FEATURE_COLS = [
    "rolling_pts_scored_5",
    "rolling_pts_allowed_5",
    "rolling_win_pct_5",
    "rolling_pts_scored_10",
    "rolling_pts_allowed_10",
    "rolling_win_pct_10",
    "home_away_pts_scored_avg",
    "home_away_pts_allowed_avg",
    "rest_days",
    "is_back_to_back",
]


def _team_key(value: Any) -> str:
    s = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    return TEAM_KEYS.get(s, s)


def _parse_matchup_id(matchup_id: str) -> tuple[str, str, pd.Timestamp]:
    parts = matchup_id.strip().split("-")
    if len(parts) != 5:
        raise ValueError(f"Bad matchup_id: {matchup_id}")
    away, home, y, m, d = parts
    return _team_key(away), _team_key(home), pd.to_datetime(f"{y}-{m}-{d}")


def _base_model(model: Any, feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_cols,
                        )
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def _player_pipeline() -> Pipeline:
    return _base_model(LinearRegression(), PLAYER_FEATURE_COLS)


def _read_player_data() -> pd.DataFrame:
    if not PLAYER_CSV.exists():
        raise FileNotFoundError(f"Missing {PLAYER_CSV}")

    df = pd.read_csv(PLAYER_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])

    if "total_rebounds" not in df.columns and "rebounds" in df.columns:
        df["total_rebounds"] = df["rebounds"]

    if "location" in df.columns:
        loc = df["location"].astype(str).str.lower()
        df["is_home"] = loc.str.contains("home").astype(int)
    elif "is_home" not in df.columns:
        df["is_home"] = 0

    df["team_key"] = df["team"].map(_team_key)
    df["opponent_key"] = df["opponent"].map(_team_key)

    missing_features = [c for c in PLAYER_FEATURE_COLS if c not in df.columns]
    missing_targets = [c for c in PLAYER_TARGETS.values() if c not in df.columns]

    if missing_features or missing_targets:
        raise ValueError(
            "Player CSV is missing columns. "
            f"Missing features: {missing_features}. "
            f"Missing targets: {missing_targets}."
        )

    return df


def _predict_player_stat_lines(
    game_date: pd.Timestamp,
    home_key: str,
    away_key: str,
    as_of_date: str,
    injury_team_abbrs: tuple[str, str],
    min_minutes: float = 10.0,
) -> list[dict]:
    df = _read_player_data()
    as_of = pd.to_datetime(as_of_date)

    train_df = df[df["game_date"] <= as_of].copy()
    if len(train_df) < 50:
        return []

    # Build the candidate roster and feature snapshots entirely from rows that
    # existed by as_of. Selecting players from the target game's box score would
    # reveal future participation even if its outcome columns were removed.
    latest_observed = train_df.sort_values(["slug", "game_date"]).drop_duplicates(
        "slug", keep="last"
    )
    roster = latest_observed[
        latest_observed["team_key"].isin([home_key, away_key])
    ][["slug", "name", "team_key"]]
    roster = roster[
        ~roster["name"].map(
            lambda player: player_is_out(
                str(player),
                injury_team_abbrs,
                parse_date(as_of_date),
            )
        )
    ]

    team_data = _read_team_data()
    snapshots = []
    for player in roster.itertuples(index=False):
        target_is_home = 1 if player.team_key == home_key else 0
        history = train_df[train_df["slug"] == player.slug].sort_values("game_date")
        venue = history[history["is_home"] == target_is_home]
        if history.empty or venue.empty:
            continue
        latest = history.iloc[-1].copy()
        for window in (5, 10):
            recent = history.tail(window)
            for source, stem in [
                ("points", "pts"),
                ("total_rebounds", "reb"),
                ("assists", "ast"),
                ("minutes", "min"),
            ]:
                latest[f"rolling_{stem}_{window}"] = recent[source].mean()
            latest[f"rolling_fg_pct_{window}"] = (
                recent["made_field_goals"].sum()
                / recent["attempted_field_goals"].sum()
                if recent["attempted_field_goals"].sum()
                else np.nan
            )
            latest[f"rolling_3p_pct_{window}"] = (
                recent["made_three_point_field_goals"].sum()
                / recent["attempted_three_point_field_goals"].sum()
                if recent["attempted_three_point_field_goals"].sum()
                else np.nan
            )
        latest["home_away_pts_avg"] = venue["points"].mean()
        latest["home_away_reb_avg"] = venue["total_rebounds"].mean()
        latest["home_away_ast_avg"] = venue["assists"].mean()
        latest["is_home"] = target_is_home
        rest = _target_rest_days(team_data, player.team_key, game_date)
        latest["rest_days"] = rest
        latest["is_back_to_back"] = int(rest == 1)
        latest["opponent"] = away_key if target_is_home else home_key
        snapshots.append(latest)

    if not snapshots:
        return []

    game_players = pd.DataFrame(snapshots)

    if "rolling_min_5" in game_players.columns:
        game_players = game_players[
            game_players["rolling_min_5"].fillna(0) >= min_minutes
        ]

    if game_players.empty:
        return []

    result = game_players[
        [
            c
            for c in ["name", "slug", "team", "opponent", "rolling_min_5"]
            if c in game_players.columns
        ]
    ].copy()

    for output_col, target_col in PLAYER_TARGETS.items():
        model = _player_pipeline()
        clean_train = train_df.dropna(subset=[target_col]).copy()
        model.fit(clean_train[PLAYER_FEATURE_COLS], clean_train[target_col])
        result[output_col] = model.predict(game_players[PLAYER_FEATURE_COLS])

    for c in ["predicted_points", "predicted_rebounds", "predicted_assists"]:
        result[c] = result[c].round(1)

    for c in ["rolling_min_5"]:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce").round(1)

    sort_cols = [c for c in ["team", "predicted_points"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols, ascending=[True, False])

    return result.replace({np.nan: None}).to_dict(orient="records")


def _first_existing_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def _read_team_data() -> pd.DataFrame:
    if not TEAM_CSV.exists():
        raise FileNotFoundError(f"Missing {TEAM_CSV}")

    df = pd.read_csv(TEAM_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team_key"] = df["team"].map(_team_key)
    df["opponent_key"] = df["opponent"].map(_team_key)

    if "location" in df.columns:
        loc = df["location"].astype(str).str.lower()
        df["_is_home"] = (
            loc.str.contains("home") | loc.isin(["h", "1", "true"])
        ).astype(int)
    elif "is_home" in df.columns:
        df["_is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("Team CSV needs a location or is_home column.")

    points = _first_existing_numeric(
        df, ["points", "pts", "team_points", "points_scored", "score", "score_for"]
    )
    opp_points = _first_existing_numeric(
        df,
        [
            "opponent_points",
            "opp_points",
            "opp_pts",
            "points_allowed",
            "score_against",
        ],
    )

    if points is not None:
        df["_points"] = points
    if opp_points is not None:
        df["_opp_points"] = opp_points

    if "won" in df.columns:
        df["_won"] = pd.to_numeric(df["won"], errors="coerce")
    elif "win" in df.columns:
        df["_won"] = pd.to_numeric(df["win"], errors="coerce")
    elif "result" in df.columns:
        df["_won"] = df["result"].astype(str).str.upper().str.startswith("W").astype(int)
    elif points is not None and opp_points is not None:
        df["_won"] = (df["_points"] > df["_opp_points"]).astype(int)
    else:
        raise ValueError("Team CSV needs won/result or points/opponent_points columns.")

    missing_features = [c for c in TEAM_FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Team CSV is missing pregame feature columns: "
            f"{missing_features}."
        )

    return df


def _target_rest_days(df: pd.DataFrame, team_key: str, game_date: pd.Timestamp) -> int:
    prior_dates = df.loc[
        (df["team_key"] == team_key) & (df["game_date"] < game_date), "game_date"
    ]
    if prior_dates.empty:
        return 7
    return int((game_date - prior_dates.max()).days)


def _team_feature_snapshot(
    df: pd.DataFrame,
    team_key: str,
    game_date: pd.Timestamp,
    as_of: pd.Timestamp,
    is_home: int,
) -> dict[str, float]:
    observed = df[(df["team_key"] == team_key) & (df["game_date"] <= as_of)].sort_values(
        "game_date"
    )
    if observed.empty:
        raise ValueError(f"No team history for {team_key} by {as_of.date()}")
    venue = observed[observed["_is_home"] == is_home]
    if venue.empty:
        venue = observed
    out: dict[str, float] = {}
    for window in (5, 10):
        recent = observed.tail(window)
        out[f"rolling_pts_scored_{window}"] = float(recent["_points"].mean())
        out[f"rolling_pts_allowed_{window}"] = float(recent["_opp_points"].mean())
        out[f"rolling_win_pct_{window}"] = float(recent["_won"].mean())
    out["home_away_pts_scored_avg"] = float(venue["_points"].mean())
    out["home_away_pts_allowed_avg"] = float(venue["_opp_points"].mean())
    rest = _target_rest_days(df, team_key, game_date)
    out["rest_days"] = float(rest)
    out["is_back_to_back"] = float(rest == 1)
    return out


def _predict_win_probability(
    game_date: pd.Timestamp,
    home_key: str,
    away_key: str,
    as_of_date: str,
) -> dict:
    df = _read_team_data()
    as_of = pd.to_datetime(as_of_date)

    home_rows = df[df["_is_home"] == 1].copy()
    away_rows = df[df["_is_home"] == 0].copy()

    base_keep = ["game_date", "team_key", "opponent_key", "_won", "_points", "_opp_points"]
    keep = [c for c in base_keep + TEAM_FEATURE_COLS if c in df.columns]

    matchups = home_rows[keep].merge(
        away_rows[keep],
        left_on=["game_date", "team_key", "opponent_key"],
        right_on=["game_date", "opponent_key", "team_key"],
        suffixes=("_home", "_away"),
    )

    train_df = matchups[matchups["game_date"] <= as_of].copy()
    if len(train_df) < 20 or train_df["_won_home"].nunique() < 2:
        return {
            "status": "unavailable",
            "reason": "Not enough prior games before the as-of date to train logistic regression.",
        }

    feature_cols = []
    home_snapshot = _team_feature_snapshot(df, home_key, game_date, as_of, 1)
    away_snapshot = _team_feature_snapshot(df, away_key, game_date, as_of, 0)
    selected_values: dict[str, float] = {}
    for c in TEAM_FEATURE_COLS:
        h = f"{c}_home"
        a = f"{c}_away"
        if h in matchups.columns and a in matchups.columns:
            d = f"{c}_diff"
            matchups[d] = matchups[h] - matchups[a]
            train_df[d] = train_df[h] - train_df[a]
            selected_values[h] = home_snapshot[c]
            selected_values[a] = away_snapshot[c]
            selected_values[d] = home_snapshot[c] - away_snapshot[c]
            feature_cols.extend([h, a, d])

    if not feature_cols:
        return {
            "status": "unavailable",
            "reason": "No usable numeric team features were found for logistic regression.",
        }

    model = _base_model(LogisticRegression(max_iter=1000), feature_cols)

    selected = pd.DataFrame([selected_values])
    model.fit(train_df[feature_cols], train_df["_won_home"].astype(int))
    home_prob = float(model.predict_proba(selected[feature_cols])[:, 1][0])
    away_prob = 1.0 - home_prob

    predicted_winner = home_key if home_prob >= 0.5 else away_key

    return {
        "status": "ok",
        "model": "Logistic Regression",
        "home_team": home_key,
        "away_team": away_key,
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
        "predicted_winner": predicted_winner,
        "feature_count": len(feature_cols),
        "training_rows": int(len(train_df)),
    }


def predict_model_only(matchup_id: str, as_of_date: str) -> dict:
    """Full Model-only output for the UI.

    Uses:
    - Logistic Regression for game winner / win probability
    - Linear Regression for player points, rebounds, and assists
    """
    try:
        away_key, home_key, game_date = _parse_matchup_id(matchup_id)
        injury_away, injury_home, _ = parse_source_matchup_id(matchup_id)
        as_of = pd.to_datetime(as_of_date).normalize()
        if as_of >= game_date.normalize():
            raise ValueError("as_of_date must be before the game date")

        win = _predict_win_probability(
            game_date=game_date,
            home_key=home_key,
            away_key=away_key,
            as_of_date=as_of_date,
        )

        players = _predict_player_stat_lines(
            game_date=game_date,
            home_key=home_key,
            away_key=away_key,
            as_of_date=as_of_date,
            injury_team_abbrs=(injury_away, injury_home),
        )

        status = "ok" if win.get("status") == "ok" else "partial"

        return {
            "status": status,
            "matchup_id": matchup_id,
            "game_date": game_date.strftime("%Y-%m-%d"),
            "as_of_date": as_of_date,
            "home": home_key,
            "away": away_key,
            "home_win_prob": win.get("home_win_prob"),
            "away_win_prob": win.get("away_win_prob"),
            "predicted_winner": win.get("predicted_winner"),
            "win_prediction": win,
            "player_stat_line_model": "Linear Regression",
            "player_stat_lines": players,
        }

    except Exception as exc:
        return {
            "status": "error",
            "matchup_id": matchup_id,
            "as_of_date": as_of_date,
            "error": f"{type(exc).__name__}: {exc}",
        }
