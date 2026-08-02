"""Fit the win-probability model and write its coefficients to JSON.

    python -m models.train

Train on the 2023-24 and 2024-25 seasons, test on 2025-26. The split is by
SEASON, not by a random shuffle, and that is the whole point: a random split
lets the model learn from March to predict January, which no real forecaster can
do, and it inflates accuracy by a few points that vanish the moment anyone
checks. The replay harness scores 2025-26, so 2025-26 is never trained on.

Why logistic regression and not the gradient booster sitting in sklearn: we are
scored on log loss and Brier as well as accuracy, and those punish badly
calibrated confidence. A booster on 2,455 games and eight features will fit the
training seasons harder and produce probabilities that are too sure of
themselves. Logistic regression on standardised features is well calibrated
almost by construction, which is what a probability is for. It is also
inspectable -- eight weights, printed below, that a human can argue with.

Output is JSON, not a pickle. `models/win_probability.json` is a few hundred
bytes of named numbers: it commits, it diffs, a reviewer can read the weights in
the pull request, and loading it needs no sklearn at all. A .joblib would be an
opaque binary that .gitignore already excludes, which would mean nobody could
run the agent without training first.

This is the file Sarvesh replaces. Keep the JSON's shape and every other thing
in the repo keeps working -- see models/README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.features import FEATURE_NAMES, build_season

MODEL_PATH = Path(__file__).resolve().parent / "win_probability.json"

TRAIN_SEASONS = (2024, 2025)
TEST_SEASON = 2026


def _matrix(seasons) -> tuple[np.ndarray, np.ndarray, list]:
    rows = []
    for s in seasons:
        season_rows, report = build_season(s)
        print(
            f"  {s}: {report['games']} games, "
            f"{report['games_with_injury_signal']} with injury signal"
        )
        rows.extend(season_rows)
    scored = [r for r in rows if r.home_won is not None]
    X = np.array([r.features for r in scored], dtype=float)
    y = np.array([r.home_won for r in scored], dtype=int)
    return X, y, scored


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return {
        "n": int(len(y)),
        "accuracy": round(float(((p >= 0.5) == y.astype(bool)).mean()), 4),
        "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
    }


def train(write: bool = True) -> dict:
    print(f"training seasons {TRAIN_SEASONS}:")
    X_train, y_train, _ = _matrix(TRAIN_SEASONS)
    print(f"test season {TEST_SEASON}:")
    X_test, y_test, test_rows = _matrix((TEST_SEASON,))

    scaler = StandardScaler().fit(X_train)
    # liblinear + a fixed seed so two people running this get identical weights;
    # a model whose coefficients move between runs cannot be reviewed in a diff.
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
    clf.fit(scaler.transform(X_train), y_train)

    train_p = clf.predict_proba(scaler.transform(X_train))[:, 1]
    test_p = clf.predict_proba(scaler.transform(X_test))[:, 1]

    payload = {
        "model": "logistic_regression_v1",
        "trained_by": "josh",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": [round(float(c), 6) for c in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
        "scaler_mean": [round(float(m), 6) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 6) for s in scaler.scale_],
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "metrics": {
            "train": _metrics(train_p, y_train),
            "test": _metrics(test_p, y_test),
        },
        "note": (
            "Trained on seasons strictly before the test season. Features are "
            "as-of by construction (models/features.py). Replace this file to "
            "swap in a different model -- see models/README.md."
        ),
    }

    print("\nweights (standardised, so they are comparable to each other):")
    order = sorted(
        zip(FEATURE_NAMES, payload["coefficients"]),
        key=lambda kv: -abs(kv[1]),
    )
    for name, c in order:
        print(f"  {name:<22} {c:+.4f}")
    print(f"  {'(intercept)':<22} {payload['intercept']:+.4f}")

    tr, te = payload["metrics"]["train"], payload["metrics"]["test"]
    print(
        f"\ntrain  acc {tr['accuracy']:.1%}  logloss {tr['log_loss']:.4f}  "
        f"brier {tr['brier']:.4f}  (n={tr['n']})"
    )
    print(
        f"test   acc {te['accuracy']:.1%}  logloss {te['log_loss']:.4f}  "
        f"brier {te['brier']:.4f}  (n={te['n']})"
    )
    gap = tr["accuracy"] - te["accuracy"]
    print(f"generalisation gap {gap:+.1%}" + ("  <- overfit" if gap > 0.05 else ""))

    # A model that cannot beat "the home team wins" is not a model. Print it
    # rather than assert it: a bad number is a finding worth reporting, and the
    # three-arm write-up is more honest for having it in the record.
    home_rate = float(y_test.mean())
    print(f"always-home on the same games: {home_rate:.1%}")

    if write:
        MODEL_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the win-probability model")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print metrics without writing the JSON"
    )
    args = ap.parse_args()
    train(write=not args.dry_run)


if __name__ == "__main__":
    main()
