"""Replay a finished season game by game and score the predictions.

This is the harness the PDP calls the project's primary contribution. For every
game it sets as_of to the morning before tip-off, asks the prediction tool for a
win probability using only what was knowable then, and compares the answer to
what actually happened.

Three numbers, because absolute accuracy on NBA games is close to meaningless
without a reference:

    accuracy    share of games called correctly
    log loss    punishes confident wrong answers (lower is better)
    Brier       mean squared error on the probability (lower is better)

Scored against two baselines:

    always-home   the naive prior. Home teams win ~55% of NBA games.
    Vegas         the closing moneyline, de-vigged. The real bar.

Leakage: the harness reads results ONLY to score, after the prediction is made.
The prediction path never touches game_logs -- it goes through the same gated
tools the agent uses. The Vegas line comes from odds_2026.csv, which carries no
score columns by construction (see scripts/build_2026_testset.py).

    python -m eval.replay --playoffs
    python -m eval.replay --limit 200 --out eval/results_regular.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources import get_source, parse_date  # noqa: E402
from agent.tools import build_tools  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def american_to_prob(ml: str) -> float | None:
    try:
        v = float(ml)
    except (TypeError, ValueError):
        return None
    return 100.0 / (v + 100.0) if v > 0 else abs(v) / (abs(v) + 100.0)


# Std dev of (actual margin - closing spread), fitted on all 1,322 games of
# 2025-26: sigma = 14.02, mean residual -0.25. The market is essentially
# unbiased, so the spread is a fair expected margin and only the spread needs
# converting. Fitted in-sample, which if anything makes this baseline slightly
# stronger than it deserves -- a conservative bar to clear.
MARGIN_SIGMA = 14.0


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def spread_home_prob(row: dict) -> float | None:
    """Home win probability implied by the closing spread."""
    try:
        spread = float(row.get("spread", ""))
    except (TypeError, ValueError):
        return None
    favored = (row.get("whos_favored") or "").strip().lower()
    if favored not in {"home", "away"}:
        return None
    expected_home_margin = spread if favored == "home" else -spread
    return _phi(expected_home_margin / MARGIN_SIGMA)


def vegas_home_prob(row: dict) -> float | None:
    """The market's home win probability.

    Prefers the two moneylines: their implied probabilities sum above 1.0, and
    that excess is the house edge, so normalising by the sum removes the vig.
    The 2025-26 rows carry no moneylines, only a spread, so fall back to that.
    """
    h, a = (
        american_to_prob(row.get("moneyline_home")),
        american_to_prob(row.get("moneyline_away")),
    )
    if h is None or a is None or (h + a) == 0:
        return spread_home_prob(row)
    return h / (h + a)


def metrics(preds: list[tuple[float, int]]) -> dict:
    """preds = [(predicted_home_win_prob, actual_home_win 0/1), ...]"""
    if not preds:
        return {"n": 0}
    n = len(preds)
    acc = sum((p >= 0.5) == bool(y) for p, y in preds) / n
    ll = (
        -sum(
            y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS))
            for p, y in preds
        )
        / n
    )
    brier = sum((p - y) ** 2 for p, y in preds) / n
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
    }


def load_rows(playoffs_only: bool, limit: int | None) -> tuple[list[dict], dict]:
    games_path = ROOT / "data/samples/game_logs_2026.csv"
    odds_path = ROOT / "data/samples/odds_2026.csv"
    if not games_path.exists():
        raise SystemExit(
            f"missing {games_path.name}. Run: python scripts/build_2026_testset.py"
        )

    games = list(csv.DictReader(open(games_path)))
    if playoffs_only:
        games = [g for g in games if g.get("playoffs") == "1"]
    if limit:
        games = games[:limit]

    odds = {}
    if odds_path.exists():
        odds = {r["matchup_id"]: r for r in csv.DictReader(open(odds_path))}
    return games, odds


def run(playoffs_only: bool, limit: int | None, out_path: str | None) -> dict:
    games, odds = load_rows(playoffs_only, limit)
    source = get_source("real")
    tools = {t.name: t for t in build_tools(source)}
    predict = tools["predict_win_probability"]

    model_preds, home_preds, vegas_preds, rows = [], [], [], []
    skipped = 0

    for g in games:
        tip = parse_date(g["game_date"])
        as_of = (tip - timedelta(days=1)).isoformat()
        actual = 1 if g["winner"] == g["home"] else 0

        try:
            res = json.loads(
                predict.invoke(
                    {
                        "home_abbr": g["home"],
                        "away_abbr": g["away"],
                        "as_of_date": as_of,
                    }
                )
            )
        except Exception:
            skipped += 1
            continue

        p = res.get("home_win_prob")
        if p is None:
            skipped += 1
            continue

        model_preds.append((p, actual))
        home_preds.append(
            (0.55, actual)
        )  # naive prior, not 1.0 -- log loss needs a probability

        vp = vegas_home_prob(odds.get(g["game_id"], {}))
        if vp is not None:
            vegas_preds.append((vp, actual))

        rows.append(
            {
                "matchup_id": g["game_id"],
                "as_of": as_of,
                "home": g["home"],
                "away": g["away"],
                "model_home_prob": round(p, 4),
                "vegas_home_prob": round(vp, 4) if vp is not None else "",
                "actual_home_win": actual,
                "playoffs": g.get("playoffs", "0"),
            }
        )

    report = {
        "scope": "2026 playoffs" if playoffs_only else "2025-26 season",
        "games_scored": len(model_preds),
        "skipped": skipped,
        "model": metrics(model_preds),
        "baseline_always_home": metrics(home_preds),
        "baseline_vegas": metrics(vegas_preds),
    }

    if out_path and rows:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        report["per_game_csv"] = str(p)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a season and score predictions")
    ap.add_argument(
        "--playoffs", action="store_true", help="Only the 2026 playoffs (the test set)"
    )
    ap.add_argument("--limit", type=int, help="Cap games (for a quick run)")
    ap.add_argument("--out", help="Write per-game predictions to this CSV")
    args = ap.parse_args()

    report = run(args.playoffs, args.limit, args.out)
    print(json.dumps(report, indent=2))

    m, v = report["model"], report["baseline_vegas"]
    if m.get("n") and v.get("n"):
        print(
            f"\nmodel {m['accuracy']:.1%} vs vegas {v['accuracy']:.1%} "
            f"({m['accuracy'] - v['accuracy']:+.1%})  |  "
            f"log loss {m['log_loss']:.3f} vs {v['log_loss']:.3f}"
        )


if __name__ == "__main__":
    main()
