"""Full-season mass evaluation: every 2025-26 game, $100 per model per game.

    python -m eval.mass_eval
    python -m eval.mass_eval --workbook

This is the professor-facing "run it in mass" harness. It scores:

    A                logistic regression, no market, no LLM
    D                market-aware accuracy model (D01, trained, no LLM)
    E                never-fade 70% + home-dog/rest overlay, else max EV from A
    always_home      naive 55% home baseline
    vegas_favorite   bet the closing favorite every game (the vig test)

B and C are Gemma 4 agents on the actual UI path. The workbook copies the
verified 10-game classroom sample plus any live full-season checkpoint rows.

Outcomes and prices are joined only after each pick is recorded. A/B/C still
cannot call a betting-line tool.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path

from eval.betting import fair_home_prob, load_odds, odds_for_matchup
from eval.policies import (
    d_home_win_prob,
    load_d_model,
    pick_accuracy,
    pick_d,
    pick_e,
    settle_pick,
)
from eval.replay import metrics
from models.features import FEATURE_NAMES, build_season
from models.predict import score_features
from models.train_d import TEST_SEASON

ROOT = Path(__file__).resolve().parents[1]
REST_I = FEATURE_NAMES.index("rest_diff")
DEFAULT_CSV = ROOT / "docs" / "evaluation" / "full-season-mass-eval.csv"
DEFAULT_LOCAL_CSV = ROOT / "eval" / "results_mass_eval.csv"
VERIFIED_BC = ROOT / "docs" / "evaluation" / "verified-actual-ui-results.csv"
LIVE_BC = ROOT / "eval" / "results_actual_ui_full_season.jsonl"
CONTRACT = "mass-eval-full-season-previous-day-de-v1"


def _winner(home: str, away: str, home_won: bool) -> str:
    return home if home_won else away


def _pick_name(home: str, away: str, pick_home: bool) -> str:
    return home if pick_home else away


def evaluate_season(d_spec: dict | None) -> list[dict]:
    rows, report = build_season(TEST_SEASON)
    rows = [r for r in rows if r.home_won is not None]
    odds = load_odds("2026")
    print(
        f"test set: {len(rows)} games of {report['games']} "
        f"({report['games_with_injury_signal']} with injury signal)"
    )

    out = []
    skipped = 0
    for row in rows:
        cutoff = (row.game_date - timedelta(days=1)).isoformat()
        p_a = score_features(row.features)
        odds_row = odds_for_matchup(row.away, row.home, row.game_date, odds)
        p_market = fair_home_prob(odds_row) if odds_row else None
        if p_market is None:
            skipped += 1
            continue
        p_d = d_home_win_prob(row.features, p_market, d_spec)
        home_won = bool(row.home_won)
        actual = _winner(row.home, row.away, home_won)

        arms = {
            "A": pick_accuracy(p_a),
            "D": pick_d(p_d, p_market, row.playoffs),
            "E": pick_e(p_a, p_market, rest_diff=row.features[REST_I]),
            "always_home": True,
            "vegas_favorite": p_market >= 0.5,
        }
        probs = {
            "A": p_a,
            "D": p_d,
            "E": p_a,
            "always_home": 0.55,
            "vegas_favorite": p_market,
        }

        record = {
            "eval_contract": CONTRACT,
            "game_id": row.game_id,
            "game_date": row.game_date.isoformat(),
            "playoffs": int(row.playoffs),
            "away": row.away,
            "home": row.home,
            "cutoff": cutoff,
            "actual_winner": actual,
            "actual_home_win": int(home_won),
            "p_a": round(p_a, 6),
            "p_d": round(p_d, 6),
            "p_market": round(p_market, 6),
            "d_disagrees_a": int(arms["D"] != arms["A"]),
            "e_disagrees_d": int(arms["E"] != arms["D"]),
            "e_fades_favorite": int(arms["E"] != arms["vegas_favorite"]),
            "odds_provenance": "reconstructed_closing_spread_hold_3.75pct",
            "polymarket_yes_price": "",
        }
        for name, pick_home in arms.items():
            settled = settle_pick(pick_home, home_won, p_market)
            record[f"{name}_p"] = round(probs[name], 6)
            record[f"{name}_pick"] = _pick_name(row.home, row.away, pick_home)
            record[f"{name}_correct"] = int(pick_home == home_won)
            record[f"{name}_decimal"] = round(settled["decimal_odds"], 6)
            record[f"{name}_american"] = settled["american_odds"]
            record[f"{name}_pnl"] = round(settled["net_pnl"], 4)
        out.append(record)

    if skipped:
        print(f"skipped {skipped} games with no usable closing line")
    return out


def summarize(records: list[dict], arms: list[str]) -> dict:
    summary = {}
    for arm in arms:
        n = len(records)
        correct = sum(r[f"{arm}_correct"] for r in records)
        preds = [(r[f"{arm}_p"], r["actual_home_win"]) for r in records]
        m = metrics(preds)
        # Accuracy is the pick, not p>=0.5. E's pick can fade a side it still
        # thinks is more likely, so those two would otherwise disagree.
        m["accuracy"] = round(correct / n, 4) if n else 0.0
        pnl = sum(r[f"{arm}_pnl"] for r in records)
        staked = n * 100.0
        summary[arm] = {
            **m,
            "correct": correct,
            "net_pnl": round(pnl, 2),
            "staked": staked,
            "roi_pct": round(100.0 * pnl / staked, 2) if staked else None,
            "ending_cash_if_funded": round(staked + pnl, 2),
        }
    summary["d_disagrees_a"] = sum(r["d_disagrees_a"] for r in records)
    summary["e_disagrees_d"] = sum(r["e_disagrees_d"] for r in records)
    summary["e_fades_favorite"] = sum(r["e_fades_favorite"] for r in records)
    return summary


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_verified_bc() -> list[dict]:
    if not VERIFIED_BC.exists():
        return []
    with VERIFIED_BC.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("arm") in {"B", "C"}]


def load_live_bc() -> list[dict]:
    """Full-season Gemma checkpoint, if the overnight run has started."""
    if not LIVE_BC.exists():
        return []
    rows = []
    for line in LIVE_BC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("arm") not in {"B", "C"}:
            continue
        rows.append(
            {
                "arm": rec["arm"],
                "game_id": rec.get("game_id", ""),
                "game_date": rec.get("game_date", ""),
                "away": rec.get("away", ""),
                "home": rec.get("home", ""),
                "cutoff": rec.get("cutoff", ""),
                "language_model": rec.get("language_model", ""),
                "home_win_prob": rec.get("home_win_prob", ""),
                "predicted_winner": rec.get("predicted_winner", ""),
                "actual_winner": rec.get("actual_winner", ""),
                "correct": rec.get("correct", ""),
                "stake": rec.get("stake", 100),
                "selected_decimal_odds": rec.get("reconstructed_decimal_odds", ""),
                "net_pnl": rec.get("net_pnl", ""),
                "tool_call_count": rec.get("tool_call_count", ""),
                "gate_failed": rec.get("gate_failed", ""),
            }
        )
    return rows


def print_report(summary: dict, arms: list[str]) -> None:
    print("\n" + "=" * 78)
    print(
        f"{'arm':<18}{'n':>5}{'acc':>8}{'correct':>9}"
        f"{'net P&L':>12}{'funded end':>14}{'ROI':>8}"
    )
    print("-" * 78)
    for arm in arms:
        s = summary[arm]
        print(
            f"{arm:<18}{s['n']:>5}{s['accuracy']:>7.1%}{s['correct']:>9}"
            f"{s['net_pnl']:>+12,.0f}{s['ending_cash_if_funded']:>14,.0f}"
            f"{s['roi_pct']:>+7.1f}%"
        )
    print("=" * 78)
    print(
        f"D disagrees with A on {summary['d_disagrees_a']} games; "
        f"E disagrees with D on {summary['e_disagrees_d']}; "
        f"E fades the favorite on {summary['e_fades_favorite']}."
    )
    print(
        "Funded end = $100 × games + net P&L. That is the cash you would hold "
        "if you set aside $100 before every game."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--local-out", type=Path, default=DEFAULT_LOCAL_CSV)
    ap.add_argument("--workbook", action="store_true")
    ap.add_argument(
        "--train-d",
        action="store_true",
        help="Refit Model D before scoring (needs sklearn).",
    )
    args = ap.parse_args()

    if args.train_d or load_d_model() is None:
        from models.train_d import train as train_d

        print("fitting Model D (A features + closing line, 2024-25 train / 2026 test)")
        train_d(write=True)

    d_spec = load_d_model()
    records = evaluate_season(d_spec)
    arms = ["A", "D", "E", "always_home", "vegas_favorite"]
    summary = summarize(records, arms)
    write_csv(args.out, records)
    write_csv(args.local_out, records)
    print(f"\nwrote {args.out} ({len(records)} games)")
    print(f"wrote {args.local_out}")
    print_report(summary, arms)

    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")

    if args.workbook:
        from eval.build_mass_workbook import build_workbook

        xlsx = ROOT / "docs" / "evaluation" / "NBA-Full-Season-Mass-Eval-Betting.xlsx"
        build_workbook(
            records,
            summary,
            load_verified_bc(),
            xlsx,
            live_bc=load_live_bc(),
        )
        print(f"wrote {xlsx}")


if __name__ == "__main__":
    main()
