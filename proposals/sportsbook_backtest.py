"""PROPOSAL PROTOTYPE -- not part of the shared eval/ harness.

Backtests the *current* predict_win_probability stub (the real one in
agent/tools.py, unmodified -- this script imports and calls it, it does not
reimplement it) against historical sportsbook lines, to answer Josh's
question: if we'd bet the model's edge over the market through a real
playoff run, would we have made money?

Data: data/raw/odds/primary/nba_2008-2026.csv -- historical game odds
(spread, total, moneyline both sides) plus final scores, 2008-2026,
regular season + playoffs. This file is gitignored (data/raw/ policy) and
was already sitting on disk locally as of 2026-07-13, not fetched by this
script and not yet part of the team's shared data layer. If the team wants
this as a real data source, that's a data/raw/odds/ addition for whoever
owns the data layer (Patrick/Kirtan) to review and commit deliberately --
not something this script does on its own.

This directly extends the retrieve_betting_line TODO already in
agent/tools.py (owner: Kirtan), which cites Sadovnik's 7/07 framing:
beating the market line is a better signal than beating the raw result,
because games have upsets.

Usage:
    python proposals/sportsbook_backtest.py --season 2024 --playoffs-only
    python proposals/sportsbook_backtest.py --season 2024 --edge 0.05
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.sources import get_source  # noqa: E402
from agent.teams import normalize_abbr  # noqa: E402
from agent.tools import _stub_win_probability  # noqa: E402

ODDS_CSV = REPO_ROOT / "data" / "raw" / "odds" / "primary" / "nba_2008-2026.csv"

# The odds file uses its own shorthand team codes that don't match
# agent/teams.py's ABBR_ALIASES (which only reconciles BKN/CHA/PHX). Six
# codes need an explicit map; everything else passes through
# normalize_abbr() unchanged.
ODDS_ABBR_FIX = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def odds_abbr(code: str) -> str:
    a = normalize_abbr(code)
    return ODDS_ABBR_FIX.get(a, a)


def implied_prob(american_odds: int) -> float:
    """American odds -> the market's implied win probability (includes vig)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)


def payout_per_unit(american_odds: int) -> float:
    """Profit per 1-unit stake if this side wins."""
    if american_odds > 0:
        return american_odds / 100
    return 100 / -american_odds


@dataclass
class GameResult:
    game_date: date
    matchup_id: str
    home: str
    away: str
    home_win: bool
    model_home_prob: float | None
    market_home_prob: float
    home_ml: int
    away_ml: int
    skip_reason: str | None = None


def load_games(season: int, playoffs_only: bool) -> list[dict]:
    rows = []
    with ODDS_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(r["season"]) != season:
                continue
            if playoffs_only and r["playoffs"] != "True":
                continue
            if not r["moneyline_home"] or not r["moneyline_away"]:
                continue
            rows.append(r)
    return rows


def evaluate_game(row: dict, source) -> GameResult:
    home = odds_abbr(row["home"])
    away = odds_abbr(row["away"])
    game_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
    as_of = game_date - timedelta(days=1)
    matchup_id = f"{away}-{home}-{game_date.isoformat()}"

    home_ml, away_ml = int(row["moneyline_home"]), int(row["moneyline_away"])
    home_score, away_score = int(row["score_home"]), int(row["score_away"])

    pred = _stub_win_probability(source, home, away, as_of.isoformat())
    model_prob = pred.get("home_win_prob")

    market_prob = implied_prob(home_ml)

    return GameResult(
        game_date=game_date,
        matchup_id=matchup_id,
        home=home,
        away=away,
        home_win=home_score > away_score,
        model_home_prob=model_prob,
        market_home_prob=market_prob,
        home_ml=home_ml,
        away_ml=away_ml,
        skip_reason=pred.get("error"),
    )


def backtest(results: list[GameResult], edge_threshold: float) -> dict:
    """Flat 1-unit stake, bet only when the model's edge over the market
    exceeds edge_threshold. No bet placed when the stub couldn't price the
    game (missing prior-season ratings) -- that's a skip, not a loss.
    """
    bets = 0
    wins = 0
    units = 0.0
    skipped = 0
    correct_side = 0
    scored = 0
    brier_sum = 0.0

    baseline_favorite_units = 0.0
    baseline_favorite_bets = 0

    for g in results:
        if g.model_home_prob is None:
            skipped += 1
            continue

        scored += 1
        predicted_home_win = g.model_home_prob >= 0.5
        if predicted_home_win == g.home_win:
            correct_side += 1
        brier_sum += (g.model_home_prob - (1.0 if g.home_win else 0.0)) ** 2

        # Baseline: always back the market favorite, flat stake -- what
        # "just trust Vegas" returns, for comparison.
        fav_is_home = g.market_home_prob >= 0.5
        baseline_favorite_bets += 1
        if fav_is_home == g.home_win:
            baseline_favorite_units += payout_per_unit(
                g.home_ml if fav_is_home else g.away_ml
            )
        else:
            baseline_favorite_units -= 1.0

        home_edge = g.model_home_prob - g.market_home_prob
        away_edge = (1 - g.model_home_prob) - (1 - g.market_home_prob)

        if home_edge >= edge_threshold:
            bets += 1
            if g.home_win:
                wins += 1
                units += payout_per_unit(g.home_ml)
            else:
                units -= 1.0
        elif away_edge >= edge_threshold:
            bets += 1
            if not g.home_win:
                wins += 1
                units += payout_per_unit(g.away_ml)
            else:
                units -= 1.0

    return {
        "games_total": len(results),
        "games_scored": scored,
        "games_skipped_no_rating": skipped,
        "model_accuracy": correct_side / scored if scored else None,
        "model_brier_score": brier_sum / scored if scored else None,
        "edge_bets_placed": bets,
        "edge_bets_won": wins,
        "edge_bet_win_rate": wins / bets if bets else None,
        "edge_bet_net_units": round(units, 2),
        "edge_bet_roi_pct": round(100 * units / bets, 1) if bets else None,
        "baseline_always_favorite_bets": baseline_favorite_bets,
        "baseline_always_favorite_net_units": round(baseline_favorite_units, 2),
        "baseline_always_favorite_roi_pct": (
            round(100 * baseline_favorite_units / baseline_favorite_bets, 1)
            if baseline_favorite_bets
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, default=2024, help="e.g. 2024 = 2023-24 season"
    )
    parser.add_argument("--playoffs-only", action="store_true")
    parser.add_argument(
        "--edge",
        type=float,
        default=0.05,
        help="Minimum model-vs-market edge (probability points) to place a bet",
    )
    args = parser.parse_args()

    source = get_source("real")
    rows = load_games(args.season, args.playoffs_only)
    results = [evaluate_game(r, source) for r in rows]

    report = backtest(results, args.edge)
    report["season"] = args.season
    report["playoffs_only"] = args.playoffs_only
    report["edge_threshold"] = args.edge

    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
