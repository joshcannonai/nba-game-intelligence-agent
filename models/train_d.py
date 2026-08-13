"""Fit Model D: Model A's features plus the closing-line win probability.

    python -m models.train_d

Same season split as A (train 2024+2025, test 2026). Extra features are the
market's de-vigged home win probability and its log-odds. That is allowed
here because D's job is to pick as many winners as possible given the book.

Fitted with NumPy Newton steps so this does not need sklearn at runtime.
Writes `models/win_probability_d.json`. Does not touch A's weights.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from eval.betting import fair_home_prob, load_odds, odds_for_matchup
from eval.policies import D_MODEL_PATH
from models.features import FEATURE_NAMES, build_season

# Duplicated from models.train so this file does not import sklearn.
TRAIN_SEASONS = (2024, 2025)
TEST_SEASON = 2026
D_FEATURE_NAMES = FEATURE_NAMES + ("market_home_prob", "market_logit")


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return {
        "n": int(len(y)),
        "accuracy": round(float(((p >= 0.5) == y.astype(bool)).mean()), 4),
        "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
    }


def _rows_with_market(seasons, odds) -> tuple[np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    skipped = 0
    for season in seasons:
        rows, report = build_season(season)
        print(
            f"  {season}: {report['games']} games, "
            f"{report['games_with_injury_signal']} with injury signal"
        )
        for row in rows:
            if row.home_won is None:
                continue
            odds_row = odds_for_matchup(row.away, row.home, row.game_date, odds)
            p_market = fair_home_prob(odds_row) if odds_row else None
            if p_market is None:
                skipped += 1
                continue
            p_m = min(1.0 - 1e-15, max(1e-15, p_market))
            logit = math.log(p_m / (1.0 - p_m))
            X_rows.append(tuple(row.features) + (p_market, logit))
            y_rows.append(row.home_won)
    if skipped:
        print(f"  skipped {skipped} games with no usable closing line")
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 40):
    """Standardised logistic regression with a Newton step. No sklearn."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    xs = (X - mean) / scale
    n, d = xs.shape
    design = np.column_stack([np.ones(n), xs])
    w = np.zeros(design.shape[1])
    ridge = np.zeros_like(w)
    ridge[1:] = l2
    for _ in range(max_iter):
        p = _sigmoid(design @ w)
        w_diag = p * (1.0 - p)
        hess = (design.T * w_diag) @ design / n
        hess[1:, 1:] += np.diag(ridge[1:] / n)
        grad = design.T @ (p - y) / n + ridge * w / n
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    intercept = float(w[0])
    coef = w[1:]
    return coef, intercept, mean, scale


def _predict_proba(X, coef, intercept, mean, scale) -> np.ndarray:
    xs = (X - mean) / scale
    return _sigmoid(xs @ coef + intercept)


def train(write: bool = True) -> dict:
    odds = load_odds(season=None)
    print(f"training seasons {TRAIN_SEASONS} (A features + market + logit):")
    X_train, y_train = _rows_with_market(TRAIN_SEASONS, odds)
    print(f"test season {TEST_SEASON}:")
    X_test, y_test = _rows_with_market((TEST_SEASON,), odds)

    coef, intercept, mean, scale = fit_logistic(X_train, y_train)
    train_p = _predict_proba(X_train, coef, intercept, mean, scale)
    test_p = _predict_proba(X_test, coef, intercept, mean, scale)

    payload = {
        "model": "logistic_regression_d_market_logit_v2",
        "trained_by": "josh",
        "feature_names": list(D_FEATURE_NAMES),
        "coefficients": [round(float(c), 6) for c in coef],
        "intercept": round(float(intercept), 6),
        "scaler_mean": [round(float(m), 6) for m in mean],
        "scaler_scale": [round(float(s), 6) for s in scale],
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "metrics": {
            "train": _metrics(train_p, y_train.astype(int)),
            "test": _metrics(test_p, y_test.astype(int)),
        },
        "note": (
            "Model D may use the closing line because its objective is accuracy "
            "given the market. A/B/C must not. Test season 2026 is held out."
        ),
    }

    print("\nweights (standardised):")
    order = sorted(
        zip(D_FEATURE_NAMES, payload["coefficients"]),
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

    if write:
        D_MODEL_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {D_MODEL_PATH.relative_to(D_MODEL_PATH.parents[1])}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Model D (accuracy, market-aware)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    train(write=not args.dry_run)


if __name__ == "__main__":
    main()
