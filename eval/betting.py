"""If you bet $100 on whoever each approach picked, who ends up with the most money?

Accuracy is an abstraction. This turns it into dollars, which is the question
anyone actually asks about a prediction system.

    python -m eval.betting

WHAT IS REAL AND WHAT IS DERIVED. This matters more than the result.

  REAL      the closing spread for every game of 2025-26, and the final scores.
  DERIVED   the moneyline price. Our odds source carries no moneylines from the
            2023-24 season onward -- the entire window we test on -- so a price
            has to be reconstructed from the spread.

The reconstruction: the closing spread is a fair estimate of the expected margin,
and margins scatter around it roughly normally (sigma = 14.0, fitted on all 1,322
games of 2025-26, mean residual -0.25). That gives a fair win probability, which
inverts to a fair moneyline. Real sportsbooks do not offer fair prices, so a
standard hold is applied to both sides.

So the dollar figures below are a SIMULATION AT REALISTIC PRICES, not a backtest
against quoted prices. The ranking between approaches is trustworthy -- every
approach faces identical prices on identical games. The absolute profit is only
as good as the price model.

Why it still answers the question: the vig is the bar. A system that predicts no
better than the market loses money slowly no matter how accurate it looks, and
that is the thing accuracy alone will not tell you.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.replay import MARGIN_SIGMA, _phi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STAKE = 100.0  # flat, every game
HOLD = 0.045  # 4.5% overround, typical for a two-way NBA moneyline


def fair_home_prob(row: dict) -> float | None:
    """Home win probability implied by the closing spread."""
    try:
        spread = float(row["spread"])
    except (TypeError, ValueError, KeyError):
        return None
    favored = (row.get("whos_favored") or "").strip().lower()
    if favored not in {"home", "away"}:
        return None
    expected_home_margin = spread if favored == "home" else -spread
    return _phi(expected_home_margin / MARGIN_SIGMA)


def priced(prob: float) -> float:
    """Decimal odds a book would offer on an outcome of this fair probability.

    Fair decimal odds are 1/p. The book shortens both sides so the two implied
    probabilities sum to 1 + HOLD rather than 1; that excess is its margin, and
    it is what a bettor has to out-predict before making a cent.
    """
    return 1.0 / (prob * (1.0 + HOLD))


def settle(pick_home: bool, home_won: bool, p_home: float) -> float:
    """Profit or loss on one $100 bet."""
    win = pick_home == home_won
    if not win:
        return -STAKE
    odds = priced(p_home if pick_home else 1.0 - p_home)
    return STAKE * (odds - 1.0)


def load_odds() -> dict:
    path = ROOT / "data/samples/odds_only.csv"
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("season") != "2026":
                continue
            key = (r["date"], r["away"].strip().lower(), r["home"].strip().lower())
            out[key] = r
    return out


def load_predictions(path: Path, arms: list[str]) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if all(r.get(a) not in (None, "") for a in arms)]


def simulate(rows: list[dict], arms: list[str], odds: dict) -> dict:
    """Bet STAKE on each arm's pick, every game, at reconstructed prices."""
    from agent.sources import parse_matchup_id
    from agent.teams import normalize_abbr

    # ESPN-style codes in the odds file vs Basketball-Reference codes in ours.
    alias = {
        "brk": "bkn",
        "cho": "cha",
        "gsw": "gs",
        "nop": "no",
        "nyk": "ny",
        "pho": "phx",
        "sas": "sa",
        "uta": "utah",
        "was": "wsh",
    }

    def find(game_id: str):
        away, home, d = parse_matchup_id(game_id)
        a, h = normalize_abbr(away).lower(), normalize_abbr(home).lower()
        for aa in (a, alias.get(a, a)):
            for hh in (h, alias.get(h, h)):
                row = odds.get((d.isoformat(), aa, hh))
                if row:
                    return row
        return None

    books = {a: {"pnl": 0.0, "n": 0, "won": 0} for a in arms}
    books["always home"] = {"pnl": 0.0, "n": 0, "won": 0}
    books["always favorite"] = {"pnl": 0.0, "n": 0, "won": 0}
    skipped = 0

    for r in rows:
        odds_row = find(r["game_id"])
        if not odds_row:
            skipped += 1
            continue
        p = fair_home_prob(odds_row)
        if p is None:
            skipped += 1
            continue
        home_won = int(r["actual_home_win"]) == 1

        for a in arms:
            pick_home = float(r[a]) >= 0.5
            b = books[a]
            b["pnl"] += settle(pick_home, home_won, p)
            b["n"] += 1
            b["won"] += pick_home == home_won

        for name, pick_home in (("always home", True), ("always favorite", p >= 0.5)):
            b = books[name]
            b["pnl"] += settle(pick_home, home_won, p)
            b["n"] += 1
            b["won"] += pick_home == home_won

    return {"books": books, "skipped": skipped}


LABELS = {
    "arm_A": "A — predictor only",
    "arm_B": "B — agent only",
    "arm_C": "C — agent + predictor",
    "vegas": "the closing line itself",
}


def report(title: str, res: dict, order: list[str]) -> None:
    print(f"\n{title}")
    print(
        f"  {'approach':<26}{'bets':>6}{'won':>7}{'win %':>9}{'profit':>12}{'ROI':>9}"
    )
    for key in order:
        b = res["books"].get(key)
        if not b or not b["n"]:
            continue
        staked = b["n"] * STAKE
        roi = b["pnl"] / staked * 100
        print(
            f"  {LABELS.get(key, key):<26}{b['n']:>6}{b['won']:>7}"
            f"{b['won'] / b['n'] * 100:>8.1f}%{b['pnl']:>+12,.0f}{roi:>+8.1f}%"
        )
    if res["skipped"]:
        print(f"  ({res['skipped']} games skipped — no usable closing line)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    global STAKE
    ap.add_argument("--stake", type=float, default=STAKE)
    args = ap.parse_args()
    STAKE = args.stake

    odds = load_odds()
    print(
        f"Flat ${STAKE:,.0f} per game. Prices reconstructed from the closing "
        f"spread with a {HOLD * 100:.1f}% hold."
    )
    print("Real: the spread and the final score. Derived: the moneyline price.")

    season = load_predictions(ROOT / "eval/results_three_arms_season.csv", ["arm_A"])
    if season:
        report(
            f"FULL SEASON — {len(season)} games, predictor only",
            simulate(season, ["arm_A"], odds),
            ["arm_A", "always favorite", "always home"],
        )

    paired = []
    for f in (
        "eval/results_three_arms_sample40.csv",
        "eval/results_three_arms_sample40_seed1.csv",
    ):
        paired += load_predictions(ROOT / f, ["arm_A", "arm_B", "arm_C"])
    if paired:
        report(
            f"ALL THREE ARMS — {len(paired)} paired games (before the skills layer)",
            simulate(paired, ["arm_A", "arm_B", "arm_C"], odds),
            ["arm_A", "arm_B", "arm_C", "always favorite", "always home"],
        )

    after = []
    for f in (
        "eval/results_skills_sample40.csv",
        "eval/results_skills_sample40_seed1.csv",
    ):
        after += load_predictions(ROOT / f, ["arm_A", "arm_C"])
    if after:
        report(
            f"AFTER THE SKILLS LAYER — {len(after)} paired games",
            simulate(after, ["arm_A", "arm_C"], odds),
            ["arm_A", "arm_C", "always favorite", "always home"],
        )

    print(
        "\nHow to read this. Beating the vig is a higher bar than beating the\n"
        "coin flip. An approach can call more games correctly than not and still\n"
        "lose money, because the price already reflects how likely the favourite\n"
        "was. That gap is what 'we are 2.5 points behind the market' costs in\n"
        "dollars, and it is the honest answer to whether this is worth betting on."
    )


if __name__ == "__main__":
    main()
