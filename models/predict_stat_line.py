"""The stat-line model's front door.

    predict_stat_line(player_name, matchup_id, as_of_date) -> dict

Mirrors `models/predict.py` in every respect that matters: no sklearn at inference
(scoring a ridge regression is a dot product), a JSON file of named numbers rather
than a pickle, and an `awaiting_input` envelope instead of a number when the model
file is missing. sklearn is needed to FIT this (`python -m models.train_stat_line`),
never to use it.

Features come from `source.player_features`, which is the date gate. This module
never opens a data file itself, so there is no second path to the player data and
nothing here can accidentally read past `as_of_date`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from agent.sources import STAT_LINE_FEATURE_KEYS

MODEL_PATH = Path(__file__).resolve().parent / "stat_line.json"

# What the report calls each target, so the agent does not have to know that
# rebounds live under `total_rebounds` in a box score.
LABELS = {"points": "points", "total_rebounds": "rebounds", "assists": "assists"}


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    spec = json.loads(MODEL_PATH.read_text())
    got = tuple(spec.get("feature_names", ()))
    if got != STAT_LINE_FEATURE_KEYS:
        # Positional feature order, same hazard as win_probability.json: a
        # mismatch would multiply the rest-day weight by the minutes value.
        raise ValueError(
            "models/stat_line.json was trained on a different feature set.\n"
            f"  file: {got}\n  code: {STAT_LINE_FEATURE_KEYS}\n"
            "Re-run `python -m models.train_stat_line`."
        )
    return spec


def model_available() -> bool:
    try:
        return load_model() is not None
    except ValueError:
        return False


def _score(spec: dict, target: dict, features: dict) -> float:
    total = float(target["intercept"])
    for name, coef, mean, scale in zip(
        spec["feature_names"],
        target["coefficients"],
        spec["scaler_mean"],
        spec["scaler_scale"],
    ):
        value = features.get(name)
        if value is None:
            raise KeyError(name)
        total += coef * ((float(value) - mean) / (scale or 1.0))
    return total


def predict_stat_line(
    source, player_name: str, matchup_id: str, as_of_date: str
) -> dict:
    """Projected points / rebounds / assists, or a stated reason there are none."""
    spec = load_model()
    if spec is None:
        return {
            "status": "awaiting_input",
            "tool": "predict_stat_line",
            "needs_from": "josh",
            "needs": (
                "models/stat_line.json is missing. Run "
                "`python -m models.train_stat_line`."
            ),
        }

    form = source.player_features(player_name, matchup_id, as_of_date)
    if not form.get("available"):
        return {
            "status": "unavailable",
            "tool": "predict_stat_line",
            "player": player_name,
            "matchup_id": matchup_id,
            "as_of_date": as_of_date,
            "reason": form.get("reason", "No as-of form for this player."),
        }

    features = form["features"]
    try:
        projections = {
            LABELS[target]: round(_score(spec, block, features), 1)
            for target, block in spec["targets"].items()
        }
    except KeyError as missing:
        return {
            "status": "unavailable",
            "tool": "predict_stat_line",
            "player": player_name,
            "reason": f"Feature {missing} is null for this player-game.",
        }

    test = spec["targets"]["points"]["metrics"]["test"]
    baseline = spec["targets"]["points"]["metrics"]["test_trailing_5_baseline"]
    return {
        "status": "ok",
        "model": spec["model"],
        "player": form.get("player", player_name),
        "team": form.get("team"),
        "opponent": form.get("opponent"),
        "game_date": form.get("game_date"),
        "as_of_date": as_of_date,
        "projection": projections,
        "trained_on": spec["train_seasons"],
        "tested_on": spec["test_season"],
        "points_mae": test["mae"],
        "points_mae_trailing_5_baseline": baseline["mae"],
        "caveat": (
            "Fitted on 2023-24 and validated on 2024-25, never on the season being "
            "replayed. A single-game stat line is high variance: the test mean "
            f"absolute error on points is {test['mae']:.1f}, so treat the number as "
            "a central estimate, not a fact."
        ),
    }
