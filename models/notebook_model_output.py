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


def _team_key(value: Any) -> str:
    s = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    return TEAM_KEYS.get(s, s)


def _parse_matchup_id(matchup_id: str) -> tuple[str, str, pd.Timestamp]:
    parts = matchup_id.strip().split("-")
    if len(parts) != 5:
        raise ValueError(f"Bad matchup_id: {matchup_id}")
    away, home, y, m, d = parts
    return _team_key(away), _team_key(home), pd.to_datetime(f"{y}-{m}-{d}")


def _base_model(model: Any) -> Pipeline:
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
                            None,
                        )
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def _player_pipeline() -> Pipeline:
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
                            PLAYER_FEATURE_COLS,
                        )
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


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
    min_minutes: float = 10.0,
) -> list[dict]:
    df = _read_player_data()
    as_of = pd.to_datetime(as_of_date)

    train_df = df[df["game_date"] <= as_of].copy()
    if len(train_df) < 50:
        return []

    game_players = df[
        (df["game_date"] == game_date)
        & (
            ((df["team_key"] == home_key) & (df["opponent_key"] == away_key))
            | ((df["team_key"] == away_key) & (df["opponent_key"] == home_key))
        )
    ].copy()

    if game_players.empty:
        return []

    if "rolling_min_5" in game_players.columns:
        game_players = game_players[
            game_players["rolling_min_5"].fillna(0) >= min_minutes
        ]

    if game_players.empty:
        return []

    result = game_players[
        [
            c
            for c in [
                "name",
                "slug",
                "team",
                "opponent",
                "rolling_min_5",
                "points",
                "total_rebounds",
                "assists",
            ]
            if c in game_players.columns
        ]
    ].copy()

    for output_col, target_col in PLAYER_TARGETS.items():
        model = _player_pipeline()
        clean_train = train_df.dropna(subset=[target_col]).copy()
        model.fit(clean_train[PLAYER_FEATURE_COLS], clean_train[target_col])
        result[output_col] = model.predict(game_players[PLAYER_FEATURE_COLS])

    rename_actuals = {
        "points": "actual_points",
        "total_rebounds": "actual_rebounds",
        "assists": "actual_assists",
    }
    result = result.rename(columns=rename_actuals)

    for c in ["predicted_points", "predicted_rebounds", "predicted_assists"]:
        result[c] = result[c].round(1)

    for c in ["actual_points", "actual_rebounds", "actual_assists", "rolling_min_5"]:
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
        df, ["points", "pts", "team_points", "score", "score_for"]
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

    return df


def _predict_win_probability(
    game_date: pd.Timestamp,
    home_key: str,
    away_key: str,
    as_of_date: str,
) -> dict:
    df = _read_team_data()
    as_of = pd.to_datetime(as_of_date)

    leakage_cols = {
        "points",
        "pts",
        "team_points",
        "score",
        "score_for",
        "opponent_points",
        "opp_points",
        "opp_pts",
        "points_allowed",
        "score_against",
        "won",
        "win",
        "result",
        "is_home",
        "_is_home",
        "_points",
        "_opp_points",
        "_won",
    }

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in leakage_cols and not c.startswith("_")
    ]

    home_rows = df[df["_is_home"] == 1].copy()
    away_rows = df[df["_is_home"] == 0].copy()

    base_keep = ["game_date", "team_key", "opponent_key", "_won", "_points", "_opp_points"]
    keep = [c for c in base_keep + numeric_cols if c in df.columns]
    
    matchups = home_rows[keep].merge(
        away_rows[keep],
        left_on=["game_date", "team_key", "opponent_key"],
        right_on=["game_date", "opponent_key", "team_key"],
        suffixes=("_home", "_away"),
    )

    selected = matchups[
        (matchups["game_date"] == game_date)
        & (matchups["team_key_home"] == home_key)
        & (matchups["opponent_key_home"] == away_key)
    ].copy()

    if selected.empty:
        return {
            "status": "unavailable",
            "reason": "Selected matchup was not found in team_game_stats_engineered.csv.",
        }

    train_df = matchups[matchups["game_date"] <= as_of].copy()
    if len(train_df) < 20 or train_df["_won_home"].nunique() < 2:
        return {
            "status": "unavailable",
            "reason": "Not enough prior games before the as-of date to train logistic regression.",
        }

    feature_cols = []
    for c in numeric_cols:
        h = f"{c}_home"
        a = f"{c}_away"
        if h in matchups.columns and a in matchups.columns:
            d = f"{c}_diff"
            matchups[d] = matchups[h] - matchups[a]
            train_df[d] = train_df[h] - train_df[a]
            selected[d] = selected[h] - selected[a]
            feature_cols.extend([h, a, d])

    if not feature_cols:
        return {
            "status": "unavailable",
            "reason": "No usable numeric team features were found for logistic regression.",
        }

    model = Pipeline(
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
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(train_df[feature_cols], train_df["_won_home"].astype(int))
    home_prob = float(model.predict_proba(selected[feature_cols])[:, 1][0])
    away_prob = 1.0 - home_prob

    predicted_winner = home_key if home_prob >= 0.5 else away_key

    actual_winner = None
    if "_won_home" in selected.columns and pd.notna(selected["_won_home"].iloc[0]):
        actual_winner = home_key if int(selected["_won_home"].iloc[0]) == 1 else away_key

    return {
        "status": "ok",
        "model": "Logistic Regression",
        "home_team": home_key,
        "away_team": away_key,
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
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
            "actual_winner": win.get("actual_winner"),
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