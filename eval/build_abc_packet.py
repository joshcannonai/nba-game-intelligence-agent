"""Professor packet for CECS 499 Models A, B, and C.

Full 2025-26 season, 1,322 games, previous-day cutoff. D and E are not in
this workbook. The old 10-game classroom sample is not in this file.

    PYTHONPATH=vendor python -m eval.build_abc_packet
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval._xlsx import write_xlsx
from eval.betting import priced

ROOT = Path(__file__).resolve().parents[1]
BC_JSONL = ROOT / "eval/results_actual_ui_full_season.jsonl"
MASS_CSV = ROOT / "docs/evaluation/full-season-mass-eval.csv"
XLSX_PATH = ROOT / "docs/evaluation/NBA-Actual-UI-Agent-Evaluation.xlsx"
SUMMARY_CSV = ROOT / "docs/evaluation/verified-full-season-abc-summary.csv"
GAMES_CSV = ROOT / "docs/evaluation/verified-full-season-abc-results.csv"

STAKE = 100.0
EXPECTED_N = 1322
B_EXTRA_TOOL_GAMES = (
    "NOP-BRK-2025-12-06",
    "SAS-NOP-2025-12-08",
    "BOS-BRK-2026-01-23",
    "BRK-PHO-2026-01-27",
    "ORL-CHO-2026-03-19",
)

MODEL_WIDTHS = [22, 12, 8, 8, 12, 8, 12, 8, 10, 14, 8, 12, 14, 10, 12, 12]
SUMMARY_WIDTHS = [78, 12, 12, 12, 14, 24, 12]
METH_WIDTHS = [36, 100]


def _num(value):
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and abs(number) < 1e12:
        return int(number)
    return number


def _detail_last(n: int) -> int:
    return n + 2


def _grouped(name: str, rows: list[list], widths: list[float], collapse: bool) -> dict:
    spec = {
        "name": name,
        "rows": rows,
        "freeze": 2,
        "totals_row": 2,
        "widths": widths,
    }
    n_detail = max(len(rows) - 2, 0)
    if collapse and n_detail:
        spec["group"] = (3, 2 + n_detail)
        spec["collapsed"] = True
        spec["summary_below"] = False
    return spec


def _load_bc() -> dict[str, dict[str, dict]]:
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    with BC_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            by_game[row["game_id"]][row["arm"]] = row
    return by_game


def _load_mass_a() -> dict[str, dict]:
    out = {}
    with MASS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["game_id"]] = row
    return out


def _settle(
    pick: str, home: str, actual: str, market_home: float
) -> tuple[float, float, int]:
    selected = priced(market_home if pick == home else 1.0 - market_home)
    correct = int(pick == actual)
    pnl = round(STAKE * (selected - 1.0), 2) if correct else -STAKE
    return selected, pnl, correct


def join_season() -> tuple[list[dict], dict]:
    bc = _load_bc()
    mass = _load_mass_a()
    both = {gid for gid, arms in bc.items() if "B" in arms and "C" in arms}
    missing_a = sorted(both - set(mass))
    extra_a = sorted(set(mass) - both)
    if missing_a or extra_a or len(both) != EXPECTED_N:
        raise SystemExit(
            f"Join failed: both={len(both)} missing_a={len(missing_a)} "
            f"extra_a={len(extra_a)} first_missing={missing_a[:5]}"
        )

    games = []
    tallies = {arm: {"n": 0, "correct": 0, "pnl": 0.0} for arm in ("A", "B", "C")}
    extra_b = []
    disagree = Counter()
    for gid in sorted(both, key=lambda g: (mass[g]["game_date"], g)):
        a_src = mass[gid]
        b_src = bc[gid]["B"]
        c_src = bc[gid]["C"]
        market = float(b_src["market_home_prob"])
        a_pick = a_src["A_pick"]
        a_dec, a_pnl, a_correct = _settle(
            a_pick, a_src["home"], a_src["actual_winner"], market
        )
        mass_correct = int(float(a_src["A_correct"]))
        if a_correct != mass_correct:
            raise SystemExit(
                f"A correct mismatch on {gid}: {a_correct} vs {mass_correct}"
            )
        if abs(a_dec - float(a_src["A_decimal"])) > 1e-6:
            raise SystemExit(
                f"A odds mismatch on {gid}: {a_dec} vs {a_src['A_decimal']}"
            )

        row = {
            "game_id": gid,
            "game_date": a_src["game_date"],
            "away": a_src["away"],
            "home": a_src["home"],
            "cutoff": a_src["cutoff"],
            "actual": a_src["actual_winner"],
            "A": {
                "pick": a_pick,
                "p": float(a_src["A_p"]),
                "correct": a_correct,
                "decimal": a_dec,
                "pnl": a_pnl,
                "tools": 0,
                "gate_passed": 0,
                "gate_failed": 0,
            },
            "B": _arm_from_jsonl(b_src),
            "C": _arm_from_jsonl(c_src),
        }
        games.append(row)
        for arm in ("A", "B", "C"):
            tallies[arm]["n"] += 1
            tallies[arm]["correct"] += row[arm]["correct"]
            tallies[arm]["pnl"] += row[arm]["pnl"]
        if row["B"]["tools"] != 5:
            extra_b.append(gid)
        if row["A"]["correct"] != row["C"]["correct"]:
            disagree["C_vs_A"] += 1
            if row["C"]["correct"]:
                disagree["C_beats_A"] += 1
            else:
                disagree["A_beats_C"] += 1
        else:
            disagree["C_agrees_A_outcome"] += 1
        if row["A"]["pick"] != row["C"]["pick"]:
            disagree["C_pick_diff_A"] += 1
        if row["B"]["correct"] != row["C"]["correct"]:
            disagree["C_vs_B"] += 1
            if row["C"]["correct"]:
                disagree["C_beats_B"] += 1
            else:
                disagree["B_beats_C"] += 1
        if row["B"]["pick"] != row["C"]["pick"]:
            disagree["C_pick_diff_B"] += 1

    expected = {
        "A": (871, -2720.60),
        "B": (813, -5229.85),
        "C": (860, -3175.21),
    }
    for arm, (correct, pnl) in expected.items():
        got_c = tallies[arm]["correct"]
        got_p = round(tallies[arm]["pnl"], 2)
        if got_c != correct or abs(got_p - pnl) > 0.011:
            raise SystemExit(
                f"{arm} failed verification: {got_c}/{tallies[arm]['n']} "
                f"pnl={got_p} (expected {correct}, {pnl})"
            )
    if sorted(extra_b) != sorted(B_EXTRA_TOOL_GAMES):
        raise SystemExit(f"Unexpected B extra-tool games: {extra_b}")
    stats = {"tallies": tallies, "disagree": dict(disagree), "extra_b": extra_b}
    return games, stats


def _arm_from_jsonl(src: dict) -> dict:
    expected = (
        round(STAKE * (float(src["reconstructed_decimal_odds"]) - 1.0), 2)
        if int(src["correct"])
        else -STAKE
    )
    if abs(float(src["net_pnl"]) - expected) > 0.011:
        raise SystemExit(f"P&L mismatch {src['arm']} {src['game_id']}")
    return {
        "pick": src["predicted_winner"],
        "p": float(src["home_win_prob"]),
        "correct": int(src["correct"]),
        "decimal": float(src["reconstructed_decimal_odds"]),
        "pnl": float(src["net_pnl"]),
        "tools": int(src["tool_call_count"]),
        "gate_passed": int(src.get("gate_passed") or 0),
        "gate_failed": int(src.get("gate_failed") or 0),
        "tool_calls": src.get("tool_calls") or [],
    }


def _model_rows(records: list[dict], arm: str) -> list[list]:
    header = [
        "Game",
        "Date",
        "Away",
        "Home",
        "Cutoff",
        "Actual",
        "P(home)",
        "Pick",
        "Correct",
        "Decimal odds",
        "Stake",
        "Net P&L",
        "Cumulative P&L",
        "Tool calls",
        "Gates passed",
        "Gates failed",
    ]
    n = len(records)
    last = _detail_last(n)
    totals = [
        f"=COUNTA(A3:A{last})",
        "",
        "",
        "",
        "",
        "",
        f"=AVERAGE(G3:G{last})",
        "",
        f"=SUM(I3:I{last})",
        f"=AVERAGE(J3:J{last})",
        f"=SUM(K3:K{last})",
        f"=SUM(L3:L{last})",
        f"=M{last}",
        f"=AVERAGE(N3:N{last})",
        f"=SUM(O3:O{last})",
        f"=SUM(P3:P{last})",
    ]
    rows = [header, totals]
    for i, rec in enumerate(records, start=3):
        cell = rec[arm]
        cum = f"=L{i}" if i == 3 else f"=M{i - 1}+L{i}"
        rows.append(
            [
                rec["game_id"],
                rec["game_date"],
                rec["away"],
                rec["home"],
                rec["cutoff"],
                rec["actual"],
                _num(cell["p"]),
                cell["pick"],
                _num(cell["correct"]),
                _num(cell["decimal"]),
                _num(STAKE),
                _num(cell["pnl"]),
                cum,
                _num(cell["tools"]),
                _num(cell["gate_passed"]),
                _num(cell["gate_failed"]),
            ]
        )
    return rows


def _summary_formula_line(label: str, sheet: str, excel_row: int) -> list:
    return [
        label,
        f"={sheet}!A2",
        f"={sheet}!I2",
        f'=IF(B{excel_row}=0,"",C{excel_row}/B{excel_row})',
        f"={sheet}!L2",
        f'=IF(B{excel_row}=0,"",B{excel_row}*100+E{excel_row})',
        f'=IF(B{excel_row}=0,"",E{excel_row}/(B{excel_row}*100))',
    ]


def _summary_rows(stats: dict) -> list[list]:
    blank = [""] * 7
    d = stats["disagree"]
    extra_n = len(stats["extra_b"])
    rows = [
        [
            "NBA Game Intelligence — CECS 499 Models A, B, and C",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "Full 2025-26 season, 1,322 games, previous-calendar-day cutoff. Same games "
            "for A, B, and C.",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "Accuracy = pick the winner before tip. Money = $100 on that pick at a "
            "reconstructed closing moneyline (σ=14, 3.75% hold). Those are different "
            "questions. Prices are not quoted moneylines and not tickets.",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "How to read the model sheets: row 2 is the season total. Games are "
            "collapsed. Click the + to the left of row 2, or outline button 2, to expand.",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        blank,
        [
            "Approach",
            "Games",
            "Correct",
            "Accuracy",
            "Net P&L",
            "Cash if funded $100/game",
            "ROI",
        ],
    ]
    r2_start = len(rows) + 1
    rows.append(
        _summary_formula_line(
            "Model A — logistic POST /api/predict", "Model_A", r2_start
        )
    )
    rows.append(
        _summary_formula_line(
            "Model B — Gemma 4 + 5 retrieval tools, no Model A tool",
            "Model_B",
            r2_start + 1,
        )
    )
    rows.append(
        _summary_formula_line(
            "Model C — same as B, plus predict_win_probability (Model A as a tool)",
            "Model_C",
            r2_start + 2,
        )
    )
    rows.extend(
        [
            blank,
            ["What to tell the professor", "", "", "", "", "", ""],
            [
                "Hypothesis: C beats A and B. Result: A is best of the three at 65.9% "
                "(871/1322). C at 65.1% (860/1322) beats B at 61.5% (813/1322). Giving "
                "Gemma Model A as a tool helped versus B. It did not beat the logistic.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "All three lose money at $100/game versus reconstructed vig. A is least "
                "bad. That is expected: a 3.75% hold makes betting favorites −EV even "
                "when you pick most winners.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "Prices are reconstructed from the closing spread, not quoted moneylines "
                "and not tickets.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "Models D and E (market-aware follow-ons) are not part of this CECS 499 "
                "submission.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            blank,
            ["Disagreements on the same 1,322 games", "", "", "", "", "", ""],
            [
                "C and A pick different teams",
                d.get("C_pick_diff_A", 0),
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "C correct and A wrong / A correct and C wrong",
                d.get("C_beats_A", 0),
                d.get("A_beats_C", 0),
                "",
                "",
                "",
                "",
            ],
            [
                "C and B pick different teams",
                d.get("C_pick_diff_B", 0),
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "C correct and B wrong / B correct and C wrong",
                d.get("C_beats_B", 0),
                d.get("B_beats_C", 0),
                "",
                "",
                "",
                "",
            ],
            blank,
            [
                f"Caveat: {extra_n} of 1,322 Model B games also called "
                "predict_win_probability (B is specified as retrieval-only). Listed on "
                "Methodology. Required retrieval gates still passed on every B/C game.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        ]
    )
    return rows


def _methodology_rows(stats: dict) -> list[list]:
    extra = ", ".join(stats["extra_b"])
    blocks = [
        (
            "What this workbook is",
            "CECS 499 professor packet for Models A, B, and C only. Every completed "
            "2025-26 game (N=1,322), predicted from the previous calendar day. Models D "
            "and E are not in this file.",
        ),
        (
            "The three arms",
            "A = logistic regression through POST /api/predict. B = Gemma 4 via Ollama "
            "through POST /api/run, five retrieval tools, no betting line, no Model A "
            "tool. C = the same as B plus predict_win_probability (Model A as one extra "
            "tool). A/B/C never see Vegas. C may agree or disagree with A.",
        ),
        (
            "Hypothesis and result",
            "Hypothesis: C beats A and B. Full season: A 871/1322 = 65.9% is best of "
            "three. C 860/1322 = 65.1% beats B 813/1322 = 61.5%. Giving the agent Model "
            "A helped versus retrieval-only Gemma. It did not beat the logistic.",
        ),
        (
            "Money",
            "Every arm bets $100 on its pick every game. Prices are reconstructed from "
            "the closing spread (margin sigma = 14, 3.75% hold measured on earlier "
            "quoted moneylines). Reproduce with python -m eval.betting --validate. "
            "Correlation vs older real moneylines is 0.996. These are not tickets and "
            "not quoted 2025-26 moneylines. All three arms lose money; A loses least. "
            "Hold makes favorites −EV, so beating 50% does not imply profit.",
        ),
        (
            "How Model A was scored",
            "Model A is not in the B/C jsonl. Game-level A rows come from the predictor "
            "mass-eval CSV on the same 1,322 game_ids. Decimal odds are the same "
            "reconstructed prices used on the B/C rows. Net P&L uses the B/C settlement "
            "rule: stake × (decimal − 1) when correct, else −$100, rounded to 2 decimals. "
            "That sum is −$2,720.60. The mass-eval CSV stores A P&L to 4 decimals "
            "(−$2,720.97); the 37-cent gap is rounding only.",
        ),
        (
            "Gating",
            "B and C run on local Gemma 4. Each tool is date-gated to the morning before "
            "tip-off. A B/C row is accepted only when required calls appear and every "
            "knowledge-gate receipt passes with zero post-cutoff records. Outcomes and "
            "prices are joined only after the UI returns a prediction. 0 gate failures "
            "on 1,322 B games and 1,322 C games.",
        ),
        (
            "Model B extra-tool caveat",
            f"{len(stats['extra_b'])} Model B games also called predict_win_probability, "
            f"which B is not supposed to have: {extra}. Those five still include the "
            "five required retrieval calls and passed gates. They are 5/1322 of B, not "
            "dropped from the headline. C called the predictor on every game.",
        ),
        (
            "Reproduce this packet",
            "B/C checkpoint is local/gitignored: eval/results_actual_ui_full_season.jsonl. "
            "A rows: docs/evaluation/full-season-mass-eval.csv. Then PYTHONPATH=vendor "
            "python -m eval.build_abc_packet. Do not hand the D/E betting workbook to class.",
        ),
    ]
    rows = [["Methodology", ""]]
    for title, body in blocks:
        rows.append([title, ""])
        rows.append([body, ""])
        rows.append(["", ""])
    return rows


def _games_to_export(games: list[dict]) -> list[dict]:
    out = []
    for rec in games:
        for arm in ("A", "B", "C"):
            cell = rec[arm]
            out.append(
                {
                    "arm": arm,
                    "game_id": rec["game_id"],
                    "game_date": rec["game_date"],
                    "away": rec["away"],
                    "home": rec["home"],
                    "cutoff": rec["cutoff"],
                    "pick": cell["pick"],
                    "actual": rec["actual"],
                    "correct": cell["correct"],
                    "home_win_prob": cell["p"],
                    "selected_decimal_odds": cell["decimal"],
                    "stake": STAKE,
                    "net_pnl": cell["pnl"],
                }
            )
    return out


def _write_csvs(games: list[dict], stats: dict) -> None:
    export = _games_to_export(games)
    fieldnames = [
        "arm",
        "game_id",
        "game_date",
        "away",
        "home",
        "cutoff",
        "pick",
        "actual",
        "correct",
        "home_win_prob",
        "selected_decimal_odds",
        "stake",
        "net_pnl",
    ]
    with GAMES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "arm",
                "n",
                "correct",
                "accuracy",
                "net_pnl",
                "staked",
                "roi_pct",
                "execution_path",
                "note",
            ],
        )
        writer.writeheader()
        paths = {
            "A": "POST /api/predict",
            "B": "POST /api/run SSE; Gemma 4; 5 retrieval tools",
            "C": "POST /api/run SSE; Gemma 4; 5 retrieval tools + predict_win_probability",
        }
        for arm in ("A", "B", "C"):
            t = stats["tallies"][arm]
            n = t["n"]
            pnl = round(t["pnl"], 2)
            writer.writerow(
                {
                    "arm": arm,
                    "n": n,
                    "correct": t["correct"],
                    "accuracy": round(t["correct"] / n, 6),
                    "net_pnl": pnl,
                    "staked": int(n * STAKE),
                    "roi_pct": round(100.0 * pnl / (n * STAKE), 4),
                    "execution_path": paths[arm],
                    "note": "reconstructed closing-spread moneylines; not tickets",
                }
            )


def build_workbook(games: list[dict], stats: dict, path: Path) -> Path:
    sheets = [
        {
            "name": "Summary",
            "rows": _summary_rows(stats),
            "freeze": 1,
            "widths": SUMMARY_WIDTHS,
        },
        _grouped("Model_A", _model_rows(games, "A"), MODEL_WIDTHS, True),
        _grouped("Model_B", _model_rows(games, "B"), MODEL_WIDTHS, True),
        _grouped("Model_C", _model_rows(games, "C"), MODEL_WIDTHS, True),
        {
            "name": "Methodology",
            "rows": _methodology_rows(stats),
            "freeze": 1,
            "widths": METH_WIDTHS,
        },
    ]
    return write_xlsx(path, sheets)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not BC_JSONL.exists():
        raise SystemExit(f"Missing B/C checkpoint: {BC_JSONL}")
    games, stats = join_season()
    build_workbook(games, stats, XLSX_PATH)
    _write_csvs(games, stats)
    digest = sha256_file(XLSX_PATH)
    print(f"wrote {XLSX_PATH}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {GAMES_CSV}")
    print(f"xlsx_sha256 {digest}")
    print(f"xlsx_bytes {XLSX_PATH.stat().st_size}")
    for arm in ("A", "B", "C"):
        t = stats["tallies"][arm]
        print(
            f"{arm} {t['correct']}/{t['n']} "
            f"{t['correct'] / t['n']:.4%} pnl={t['pnl']:+.2f}"
        )


if __name__ == "__main__":
    main()
