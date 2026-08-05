"""Fit the points/rebounds/assists model and write its coefficients to JSON.

    python -m models.train_stat_line

This is the "statistics" half of the report we pitched, and the model behind
`predict_stat_line`. It follows `models/train.py` in every structural way: split by
SEASON not at random, standardised features, ridge coefficients written to readable
JSON rather than a pickle, so the weights land in a diff a teammate can argue with.

THE SPLIT, AND WHY IT IS NOT 2025-26. The agent replays 2025-26. A stat-line model
fitted on 2025-26 and then asked to project a 2025-26 game would be reading the
season it is being graded on -- the same failure as serving a current-season team
rating (agent/sources.py, rule 2). So this trains on 2023-24, tests on 2024-25, and
never touches 2025-26 at all. 2025-26 is inference data only, served through the
date gate.

THE BASELINE THAT MATTERS. A stat-line model has an obvious competitor: just predict
the player's own trailing 5-game average. That is what a person does in their head,
it costs nothing, and any model that cannot beat it is not worth shipping. It is
printed next to the model's error every run, the same way `models/train.py` prints
always-pick-home. If the model loses, that is a finding to report, not to bury.

WHY RIDGE. Sixteen correlated features -- a 5-game and a 10-game average of the same
quantity are nearly the same column -- make plain OLS coefficients unstable and
unreadable, which defeats the point of writing them to a reviewable JSON. A small L2
penalty stabilises them without turning the model into a black box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.stat_line_features import (
    FEATURE_NAMES,
    TARGETS,
    engineer,
    load_box_scores,
    usable,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__file__).resolve().parent / "stat_line.json"
PRIOR_DIR = REPO_ROOT / "data" / "raw" / "player_box_scores_prior"

TRAIN_SEASON = 2024  # 2023-24
TEST_SEASON = 2025  # 2024-25
ALPHA = 1.0


def season_of(game_date: str) -> int:
    """NBA seasons straddle the new year; label them by the year they end in."""
    stamp = pd.Timestamp(game_date)
    return stamp.year + 1 if stamp.month >= 10 else stamp.year


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    resid = pred - actual
    ss_res = float((resid**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "n": int(len(actual)),
        "mae": round(float(np.abs(resid).mean()), 4),
        "rmse": round(float(np.sqrt((resid**2).mean())), 4),
        "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot else None,
    }


def train(write: bool = True) -> dict:
    print(f"loading box scores from {PRIOR_DIR.relative_to(REPO_ROOT)}")
    raw = load_box_scores(PRIOR_DIR)
    print(f"  {len(raw)} raw player-games")

    frame = usable(engineer(raw))
    frame = frame.assign(season=frame["game_date"].map(season_of))

    train_df = frame[frame["season"] == TRAIN_SEASON]
    test_df = frame[frame["season"] == TEST_SEASON]
    print(f"  train {TRAIN_SEASON}: {len(train_df)} usable player-games")
    print(f"  test  {TEST_SEASON}: {len(test_df)} usable player-games")
    if train_df.empty or test_df.empty:
        raise SystemExit(
            "one of the seasons is empty -- check the scrape covered both ranges"
        )

    X_train = train_df[list(FEATURE_NAMES)].to_numpy(dtype=float)
    X_test = test_df[list(FEATURE_NAMES)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

    # The naive competitor, per target: the player's own trailing 5-game average.
    baseline_feature = {
        "points": "rolling_pts_5",
        "total_rebounds": "rolling_reb_5",
        "assists": "rolling_ast_5",
    }

    targets_payload = {}
    for target in TARGETS:
        y_train = train_df[target].to_numpy(dtype=float)
        y_test = test_df[target].to_numpy(dtype=float)

        model = Ridge(alpha=ALPHA, random_state=0)
        model.fit(Xs_train, y_train)

        test_pred = model.predict(Xs_test)
        naive_pred = test_df[baseline_feature[target]].to_numpy(dtype=float)

        targets_payload[target] = {
            "coefficients": [round(float(c), 6) for c in model.coef_],
            "intercept": round(float(model.intercept_), 6),
            "metrics": {
                "train": _metrics(model.predict(Xs_train), y_train),
                "test": _metrics(test_pred, y_test),
                "test_trailing_5_baseline": _metrics(naive_pred, y_test),
            },
        }

        te = targets_payload[target]["metrics"]["test"]
        base = targets_payload[target]["metrics"]["test_trailing_5_baseline"]
        verdict = "beats" if te["mae"] < base["mae"] else "LOSES TO"
        print(
            f"\n{target}: test MAE {te['mae']:.3f}  rmse {te['rmse']:.3f}  "
            f"r2 {te['r2']:.3f}"
        )
        print(
            f"  {verdict} trailing-5 baseline (MAE {base['mae']:.3f}), "
            f"delta {te['mae'] - base['mae']:+.3f}"
        )
        top = sorted(
            zip(FEATURE_NAMES, targets_payload[target]["coefficients"]),
            key=lambda kv: -abs(kv[1]),
        )[:4]
        print("  top weights: " + ", ".join(f"{n} {c:+.3f}" for n, c in top))

    payload = {
        "model": "ridge_stat_line_v1",
        "trained_by": "josh",
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [round(float(m), 6) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 6) for s in scaler.scale_],
        "alpha": ALPHA,
        "train_seasons": [TRAIN_SEASON],
        "test_season": TEST_SEASON,
        "targets": targets_payload,
        "note": (
            "Trained on 2023-24, tested on 2024-25, and never fitted on 2025-26 -- "
            "the season the replay harness scores. Features are as-of by "
            "construction (models/stat_line_features.py). Replace this file to swap "
            "in a different model; keep the JSON shape and the tool keeps working."
        ),
    }

    if write:
        MODEL_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {MODEL_PATH.relative_to(REPO_ROOT)}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the stat-line model")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print metrics, write nothing"
    )
    args = ap.parse_args()
    train(write=not args.dry_run)


if __name__ == "__main__":
    main()
