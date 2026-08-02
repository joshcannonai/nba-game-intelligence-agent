"""The model's front door. One function, one shape, everything else reads it.

    predict(home_abbr, away_abbr, as_of_date) -> dict with home_win_prob

`eval/three_arms.py` calls it for arm A, `agent/tools.py` hands it to the agent
for arm C, and the Streamlit report calls it for one game. None of them know or
care what is behind it, which is the point: swapping the model is a change to
this file and nothing else.

Deliberately no sklearn import. Scoring a logistic regression is a dot product,
so the agent, the harness and the UI all load a few hundred bytes of JSON
instead of a training stack. sklearn is needed to FIT the model
(`python -m models.train`), never to USE it.

When `win_probability.json` is missing this returns the same
`status: awaiting_input` envelope the placeholder tools use, rather than a
number. A missing model is a fact worth reporting; a guess dressed as a
prediction is how the three-arm comparison quietly stops meaning anything.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from agent.sources import parse_date
from agent.teams import normalize_abbr
from models.features import FEATURE_NAMES, live_features

MODEL_PATH = Path(__file__).resolve().parent / "win_probability.json"


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    spec = json.loads(MODEL_PATH.read_text())
    got = tuple(spec.get("feature_names", ()))
    if got != FEATURE_NAMES:
        # Feature order is positional in the JSON. A mismatch means the file was
        # trained against a different feature list, and silently scoring it would
        # multiply the rest-day weight by the injury value. Refuse instead.
        raise ValueError(
            "models/win_probability.json was trained on a different feature set.\n"
            f"  file: {got}\n  code: {FEATURE_NAMES}\n"
            "Re-run `python -m models.train`."
        )
    return spec


def model_available() -> bool:
    try:
        return load_model() is not None
    except ValueError:
        return False


def score_features(features) -> float:
    """Probability the home team wins, from an already-built feature vector.

    Split out from `predict` so the replay harness can score thousands of
    pre-computed rows without rebuilding each one from the season log.
    """
    spec = load_model()
    if spec is None:
        raise FileNotFoundError("no model file")
    z = spec["intercept"]
    for x, mean, scale, coef in zip(
        features, spec["scaler_mean"], spec["scaler_scale"], spec["coefficients"]
    ):
        z += coef * ((x - mean) / (scale or 1.0))
    return 1.0 / (1.0 + math.exp(-z))


def predict(home_abbr: str, away_abbr: str, as_of_date: str) -> dict:
    """Win probability for one game, using only what was known on as_of_date.

    Args:
        home_abbr: Home team abbreviation, e.g. BOS.
        away_abbr: Away team abbreviation.
        as_of_date: ISO date. Nothing after this may inform the answer.

    Returns a dict carrying at minimum `home_win_prob` (float 0-1) or, when the
    model file is absent, `status: awaiting_input`.
    """
    home, away = normalize_abbr(home_abbr), normalize_abbr(away_abbr)

    try:
        spec = load_model()
    except ValueError as exc:
        return {
            "status": "error",
            "model": "unusable",
            "home": home,
            "away": away,
            "as_of_date": as_of_date,
            "error": str(exc),
        }

    if spec is None:
        return {
            "status": "awaiting_input",
            "tool": "predict_win_probability",
            "needs_from": "Sarvesh (models) -- or run `python -m models.train`",
            "needs": (
                "models/win_probability.json. The interface and a working "
                "baseline both exist; this message means the file was deleted "
                "or never generated."
            ),
            "home": home,
            "away": away,
            "as_of_date": as_of_date,
        }

    as_of = parse_date(as_of_date)
    features = live_features(home, away, as_of)
    p = score_features(features)

    return {
        "status": "ok",
        "model": spec["model"],
        "trained_by": spec.get("trained_by", "unknown"),
        "home": home,
        "away": away,
        "as_of_date": as_of_date,
        "home_win_prob": round(p, 4),
        "away_win_prob": round(1.0 - p, 4),
        "features": {n: round(v, 3) for n, v in zip(FEATURE_NAMES, features)},
        "trained_on_seasons": spec.get("train_seasons"),
        "holdout_accuracy": spec.get("metrics", {}).get("test", {}).get("accuracy"),
    }
