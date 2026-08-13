"""Build the professor-facing full-season mass-eval workbook.

Summary numbers are Excel formulas pointing at each model sheet's totals row,
not pre-written constants. Game rows are grouped and collapsed with Excel's
native outline: click the + at the left of row 2, or the outline "2" button,
to expand every game.

    python -m eval.mass_eval --workbook
"""

from __future__ import annotations

from pathlib import Path

from eval._xlsx import write_xlsx

ARMS = ("A", "D", "E", "always_home", "vegas_favorite")
SHEET_NAMES = {
    "A": "Model_A",
    "D": "Model_D",
    "E": "Model_E",
    "always_home": "Always_Home",
    "vegas_favorite": "Vegas_Favorites",
}

MODEL_WIDTHS = [24, 12, 10, 8, 8, 14, 12, 11, 10, 10, 12, 12, 12, 14, 14]
GAMES_WIDTHS = [
    22,
    12,
    10,
    8,
    8,
    12,
    10,
    8,
    8,
    8,
    10,
    10,
    10,
    10,
    12,
    12,
    14,
    10,
    10,
    10,
    10,
    10,
    10,
]
BC_WIDTHS = [8, 22, 12, 8, 8, 12, 22, 12, 12, 12, 10, 8, 12, 12, 12, 10]


def _detail_last(n: int) -> int:
    """Last Excel row when row 1 is header, row 2 is totals, games start at 3."""
    return n + 2


def _model_rows(arm: str, records: list[dict]) -> list[list]:
    header = [
        "Game",
        "Date",
        "Playoffs",
        "Away",
        "Home",
        "Knowledge cutoff",
        "Actual winner",
        "P(home)",
        "Pick",
        "Correct",
        "Moneyline",
        "Decimal odds",
        "Net P&L",
        "Cumulative P&L",
        "Funded cash",
    ]
    last = _detail_last(len(records))
    totals = [
        f"=COUNTA(A3:A{last})",
        "",
        f"=SUM(C3:C{last})",
        "",
        "",
        "",
        "",
        f"=AVERAGE(H3:H{last})",
        "",
        f"=SUM(J3:J{last})",
        "",
        f"=AVERAGE(L3:L{last})",
        f"=SUM(M3:M{last})",
        f"=N{last}",
        f"=O{last}",
    ]
    rows = [header, totals]
    for i, row in enumerate(records, start=3):
        cum = f"=M{i}" if i == 3 else f"=N{i - 1}+M{i}"
        funded = f"={(i - 2) * 100}+N{i}"
        rows.append(
            [
                row["game_id"],
                row["game_date"],
                row["playoffs"],
                row["away"],
                row["home"],
                row["cutoff"],
                row["actual_winner"],
                row[f"{arm}_p"],
                row[f"{arm}_pick"],
                row[f"{arm}_correct"],
                row[f"{arm}_american"],
                row[f"{arm}_decimal"],
                row[f"{arm}_pnl"],
                cum,
                funded,
            ]
        )
    return rows


def _games_rows(records: list[dict]) -> list[list]:
    header = [
        "Game",
        "Date",
        "Playoffs",
        "Away",
        "Home",
        "Cutoff",
        "Actual",
        "A pick",
        "D pick",
        "E pick",
        "Favorite",
        "A correct",
        "D correct",
        "E correct",
        "D disagrees A",
        "E disagrees D",
        "E fades favorite",
        "A P&L",
        "D P&L",
        "E P&L",
        "P(A)",
        "P(D)",
        "P(market)",
    ]
    last = _detail_last(len(records))
    totals = [
        f"=COUNTA(A3:A{last})",
        "",
        f"=SUM(C3:C{last})",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"=SUM(L3:L{last})",
        f"=SUM(M3:M{last})",
        f"=SUM(N3:N{last})",
        f"=SUM(O3:O{last})",
        f"=SUM(P3:P{last})",
        f"=SUM(Q3:Q{last})",
        f"=SUM(R3:R{last})",
        f"=SUM(S3:S{last})",
        f"=SUM(T3:T{last})",
        f"=AVERAGE(U3:U{last})",
        f"=AVERAGE(V3:V{last})",
        f"=AVERAGE(W3:W{last})",
    ]
    rows = [header, totals]
    for row in records:
        rows.append(
            [
                row["game_id"],
                row["game_date"],
                row["playoffs"],
                row["away"],
                row["home"],
                row["cutoff"],
                row["actual_winner"],
                row["A_pick"],
                row["D_pick"],
                row["E_pick"],
                row["vegas_favorite_pick"],
                row["A_correct"],
                row["D_correct"],
                row["E_correct"],
                row["d_disagrees_a"],
                row["e_disagrees_d"],
                row["e_fades_favorite"],
                row["A_pnl"],
                row["D_pnl"],
                row["E_pnl"],
                row["p_a"],
                row["p_d"],
                row["p_market"],
            ]
        )
    return rows


def _bc_rows(bc_rows: list[dict]) -> list[list]:
    header = [
        "arm",
        "game_id",
        "game_date",
        "away",
        "home",
        "cutoff",
        "language_model",
        "home_win_prob",
        "predicted_winner",
        "actual_winner",
        "correct",
        "stake",
        "selected_decimal_odds",
        "net_pnl",
        "tool_call_count",
        "gate_failed",
    ]
    n = len(bc_rows)
    last = _detail_last(n) if n else 2
    totals = [
        "",
        f"=COUNTA(B3:B{last})" if n else 0,
        "",
        "",
        "",
        "",
        "",
        f"=AVERAGE(H3:H{last})" if n else "",
        "",
        "",
        f"=SUM(K3:K{last})" if n else 0,
        f"=SUM(L3:L{last})" if n else 0,
        f"=AVERAGE(M3:M{last})" if n else "",
        f"=SUM(N3:N{last})" if n else 0,
        f"=AVERAGE(O3:O{last})" if n else "",
        f"=SUM(P3:P{last})" if n else 0,
    ]
    rows = [header, totals]
    for row in bc_rows:
        values = []
        for key in header:
            raw = row.get(key, "")
            if key in {
                "home_win_prob",
                "correct",
                "stake",
                "selected_decimal_odds",
                "net_pnl",
                "tool_call_count",
                "gate_failed",
            }:
                try:
                    raw = float(raw)
                except (TypeError, ValueError):
                    pass
            values.append(raw)
        rows.append(values)
    return rows


def _grouped(name: str, rows: list[list], widths: list[float]) -> dict:
    n_detail = max(len(rows) - 2, 0)
    spec = {
        "name": name,
        "rows": rows,
        "freeze": 2,
        "totals_row": 2,
        "widths": widths,
    }
    if n_detail:
        spec["group"] = (3, 2 + n_detail)
        spec["collapsed"] = True
        spec["summary_below"] = False
    return spec


def _summary_rows(n: int) -> list[list]:
    blank = [""] * 7
    rows = [
        [
            "NBA Game Intelligence — 2025-26 season, $100 per model per game",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "Accuracy = did you pick the winner before tip. Money = $100 on that "
            "pick at a reconstructed closing moneyline. Those are different "
            "questions: Vegas gets the most games right and still loses to the vig.",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "How to read a model sheet: row 2 is the season total. The individual "
            "games are collapsed. Click the + to the left of row 2, or the outline "
            "button 2 in the top-left margin, to expand every game. That is Excel's "
            "native grouping, not a filter.",
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
    order = [
        ("A", "Model A — no Vegas, pick the winner", "model"),
        ("B", "Model B — Gemma 4, retrieval only (full season, live)", "bc_live"),
        ("C", "Model C — Gemma 4 + Model A as a tool (full season, live)", "bc_live"),
        ("D", "Model D — may see Vegas, pick as many winners as possible", "model"),
        ("E", "Model E — may see Vegas, try to make money vs the price", "model"),
        ("always_home", "Always home — naive 55% baseline", "model"),
        (
            "vegas_favorite",
            "Vegas favorites — $100 on the closing favorite every game",
            "model",
        ),
    ]
    start = 6
    for i, (arm, label, kind) in enumerate(order):
        r = start + i
        if kind == "bc_live":
            sheet = "Gemma_BC"
            rows.append(
                [
                    label,
                    f'=COUNTIF({sheet}!A:A,"{arm}")',
                    f'=SUMIF({sheet}!A:A,"{arm}",{sheet}!K:K)',
                    f'=IF(B{r}=0,"",C{r}/B{r})',
                    f'=SUMIF({sheet}!A:A,"{arm}",{sheet}!N:N)',
                    f'=IF(B{r}=0,"",B{r}*100+E{r})',
                    f'=IF(B{r}=0,"",E{r}/(B{r}*100))',
                ]
            )
            continue
        sheet = SHEET_NAMES[arm]
        rows.append(
            [
                label,
                f"={sheet}!A2",
                f"={sheet}!J2",
                f'=IF(B{r}=0,"",C{r}/B{r})',
                f"={sheet}!M2",
                f'=IF(B{r}=0,"",B{r}*100+E{r})',
                f'=IF(B{r}=0,"",E{r}/(B{r}*100))',
            ]
        )
    rows.extend(
        [
            blank,
            [
                "How to read the money: Net P&L is profit versus $0. "
                "Cash if funded is $100 × games + P&L. A 1,322-game season is a "
                "$132,200 pool if you set aside $100 before every game.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            blank,
            [
                "Classroom 10-game Gemma sample (fixed 2026-04-05 cutoff). Not a season.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "Model",
                "Games",
                "Correct",
                "Accuracy",
                "Net P&L",
                "Cash if funded",
                "ROI",
            ],
        ]
    )

    def _bc_line(label: str, sheet: str, arm: str, excel_row: int) -> list:
        games = f'=COUNTIF({sheet}!A:A,"{arm}")'
        correct = f'=SUMIF({sheet}!A:A,"{arm}",{sheet}!K:K)'
        pnl = f'=SUMIF({sheet}!A:A,"{arm}",{sheet}!N:N)'
        return [
            label,
            games,
            correct,
            f'=IF(B{excel_row}=0,"",C{excel_row}/B{excel_row})',
            pnl,
            f'=IF(B{excel_row}=0,"",B{excel_row}*100+E{excel_row})',
            f'=IF(B{excel_row}=0,"",E{excel_row}/(B{excel_row}*100))',
        ]

    b10 = len(rows) + 1
    c10 = b10 + 1
    rows.append(
        _bc_line(
            "Model B — classroom 10-game sample",
            "Gemma_BC_10",
            "B",
            b10,
        )
    )
    rows.append(
        _bc_line(
            "Model C — classroom 10-game sample",
            "Gemma_BC_10",
            "C",
            c10,
        )
    )
    rows.extend(
        [
            blank,
            ["Which machine would you follow next season?", "", "", "", "", "", ""],
            [
                "Winners: Model D is essentially Vegas. The Live_Tweaks sheet is 30 "
                "sequential passes. Adding log-odds of the close raised D to 69.1%. "
                "Nothing gated reached 100%. Pass 31 (CHEAT) uses the winner and hits "
                "100% — that is not a model.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "Money: the live E hill-climb kept never-fade-70% at a better 2026 "
                "P&L than betting A. That is a 2026 peek. Vegas favorites still lose "
                "to the vig. Open Live_Tweaks for each pass: win rate, what changed, "
                "and why the next tweak.",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            blank,
            [
                "Disagreement checks (from the Games sheet totals row)",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            ["D disagrees with A", "=Games!O2", "", "", "", "", ""],
            ["E disagrees with D", "=Games!P2", "", "", "", "", ""],
            ["E fades the favorite", "=Games!Q2", "", "", "", "", ""],
        ]
    )
    return rows


def _methodology_rows() -> list[list]:
    blocks = [
        (
            "What this workbook is",
            "Every completed 2025-26 game, predicted from the previous calendar "
            "day, then settled at a reconstructed closing moneyline. Accuracy = "
            "did you pick the team that won. Money = $100 on that pick every game. "
            "Vegas favorites get the most games right and still lose money to the vig.",
        ),
        (
            "Vegas favorites vs always home",
            "Vegas favorites means: bet $100 on whoever the closing line says is "
            "more likely to win (the minus-money side). Always home is the naive "
            "55% rule. They are not the same. Always-favorite was removed because "
            "it was identical to Vegas favorites.",
        ),
        (
            "Models B and C",
            "Gemma 4 via Ollama, actual UI path. B has retrieval tools only. C "
            "is B plus Model A's predictor as one extra tool, and may disagree. "
            "Gemma_BC_10 is the classroom 10-game sample. Gemma_BC is the live "
            "full-season checkpoint. Game rows on those sheets are collapsed; "
            "click + to expand.",
        ),
        (
            "How D and E were chosen (15 tests each)",
            "Fifteen D recipes and fifteen E recipes were frozen, fit or configured "
            "on 2024-25 only, then scored on 2025-26. We select the winner on 2025 "
            "(D: accuracy, E: ROI) so 2026 is a holdout — that is the number you "
            "would actually follow next season. Injury weight 0x/0.5x/2x/3x did "
            "not change D's picks once the market feature was present.",
        ),
        (
            "Prices",
            "2025-26 odds are spreads, not quoted moneylines. Prices are inverted "
            "from the close with a 3.75% hold (corr 0.996 vs older real moneylines). "
            "python -m eval.betting --validate. Polymarket closes are not in the repo.",
        ),
        (
            "Reproduce",
            "python -m eval.search_de && python -m eval.mass_eval --workbook",
        ),
    ]
    rows = [["Methodology", ""]]
    for title, body in blocks:
        rows.append([title, ""])
        rows.append([body, ""])
        rows.append(["", ""])
    return rows


def _anticheat_rows() -> list[list]:
    return [
        ["How we stop the models from cheating", ""],
        [
            "The 2025-26 season already happened. An online LLM might remember who won. "
            "So B and C run on local Gemma 4 (cutoff ~Jan 2025), every tool is date-gated "
            "to the morning before tip-off, and the scorer reads results only after the pick.",
            "",
        ],
        ["Gate", "What it blocks"],
        [
            "No betting-line tool on A/B/C",
            "The market is the thing we grade against. Watching the agent quote ORL -5.5 "
            "is why retrieve_betting_line was removed. tests/test_date_gating.py keeps it gone.",
        ],
        [
            "D/E may see the close; they are a different experiment",
            "D asks: if you are allowed to look at Vegas, can you pick more winners? "
            "E asks: can you make money? Those recipes never write back into A/B/C.",
        ],
        [
            "Season split, not a shuffle",
            "A and D train on 2024-25, test 2026. The 15-recipe search selects on 2025, "
            "reports 2026. Tuning on 2026 would be the cheat this sheet exists to avoid.",
        ],
        [
            "Injury dates stop at yesterday",
            "A same-day IL transaction might be after tip-off, so replay injury reads "
            "stop at the previous calendar day.",
        ],
        [
            "B is not a stub and C is not a copy of A",
            "B/C construct ChatOllama(gemma4). C's skill says Model A is peer evidence "
            "and may disagree. The 10-game sample includes a C disagreement (IND-BRK).",
        ],
        [
            "Outcomes after picks",
            "eval/mass_eval.py and eval/ui_agent_eval.py join the winner and the price "
            "only after the model has returned.",
        ],
    ]


def _search_rows() -> list[list]:
    import json

    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "evaluation"
        / "de-search-log.json"
    )
    if not path.exists():
        return [["No search log yet. Run python -m eval.search_de."]]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        [
            "30 tests: 15 for Model D (pick winners) and 15 for Model E (make money). "
            "Each recipe is frozen on 2024-25. We select the winner on 2025 and report "
            "2026, so this is the number you would follow next season — not the best "
            "score you can cherry-pick after seeing 2026.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "D's job is accuracy. E's job is ROI. A recipe that wins 2026 but lost "
            "2025 is interesting, and it is not the one you would have shipped.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "id",
            "what it tries",
            "2025 accuracy",
            "2025 ROI %",
            "2026 accuracy",
            "2026 net P&L",
            "2026 ROI %",
            "selected on 2025",
        ],
    ]
    for arm in ("D", "E"):
        chosen = payload[arm]["selected_on_2025"]
        rows.append([f"{arm} — 15 recipes", "", "", "", "", "", "", ""])
        for rec in payload[arm]["runs"]:
            t, s = rec["tune_2025"], rec["test_2026"]
            rows.append(
                [
                    rec["id"],
                    rec["goal"],
                    t["accuracy"],
                    t["roi_pct"],
                    s["accuracy"],
                    s["net_pnl"],
                    s["roi_pct"],
                    "YES" if rec["id"] == chosen else "",
                ]
            )
    return rows


def _live_tweak_rows() -> list[list]:
    import json

    root = Path(__file__).resolve().parents[1]
    gemma = root / "docs" / "evaluation" / "gemma-de-live-log.json"
    cpu = root / "docs" / "evaluation" / "live-tweak-log.json"
    path = gemma if gemma.exists() else cpu
    if not path.exists():
        return [["No live-tweak log yet. Run python -m eval.gemma_de."]]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path == gemma:
        note = (
            "30 live GEMMA 4 passes for D and E. Each row is one prompt tweak, the "
            "sample win rate after that tweak, whether we kept it, and what we will "
            "change next. D maximizes winners; E maximizes money. Both may call "
            "retrieve_betting_line. A/B/C still cannot. Sample is previous-day gated. "
            f"Games: {payload.get('n_games')}. Vegas on this sample: "
            f"{payload.get('vegas_accuracy')}."
        )
        best_line = (
            f"Best Gemma D win rate: {payload.get('best_d_win_rate')}. "
            f"Kept D rules: {payload.get('kept_d_rules')}. "
            f"Best Gemma E P&L: {payload.get('best_e_pnl')} "
            f"(win rate {payload.get('best_e_win_rate')})."
        )
    else:
        note = (
            "30 live CPU passes. Each row is one tweak, the win rate after that tweak, "
            "whether we kept it, and what we will change next and why. D hill-climbs "
            "2026 win rate with date-gated features only. E hill-climbs 2026 ROI. "
            "Pass 31 is CHEAT (uses the winner) so the 100% row has a name — it is "
            "not a candidate. B and C are the Gemma full-season run, not these loops."
        )
        best_line = (
            f"Best gated D win rate: {payload.get('best_d_win_rate_2026')}. "
            f"Kept extras: {', '.join(payload.get('kept_d_extras') or ['(none)'])}. "
            f"Best gated E ROI: {payload.get('best_e_roi_2026')}% "
            f"(win rate {payload.get('best_e_win_rate_2026')})."
        )
    rows = [
        [
            note,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            best_line,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "Pass",
            "Arm",
            "What I tweaked",
            "Why I tried it",
            "Win rate after this pass (2026)",
            "Correct / 1322",
            "Net P&L",
            "Kept?",
            "Running best win rate",
            "What I will change next, and why",
        ],
    ]
    for rec in payload.get("passes", []):
        rows.append(
            [
                rec["pass"],
                rec["arm"],
                rec["tweak"],
                rec["why_this_pass"],
                rec["win_rate_2026"],
                rec["correct_2026"],
                rec["pnl_2026"],
                "YES" if rec["kept"] else "no — reverted",
                rec["running_best_win_rate_2026"],
                rec["next_change_and_why"],
            ]
        )
    return rows


def build_workbook(
    records: list[dict],
    summary: dict,
    bc_rows: list[dict],
    path: Path,
    live_bc: list[dict] | None = None,
) -> Path:
    n = len(records)
    sheets = [
        {
            "name": "Summary",
            "rows": _summary_rows(n),
            "freeze": 1,
            "widths": [72, 12, 12, 12, 14, 22, 10],
        },
        _grouped("Games", _games_rows(records), GAMES_WIDTHS),
    ]
    for arm in ARMS:
        sheets.append(
            _grouped(SHEET_NAMES[arm], _model_rows(arm, records), MODEL_WIDTHS)
        )
    sheets.extend(
        [
            _grouped("Gemma_BC_10", _bc_rows(bc_rows), BC_WIDTHS),
            _grouped("Gemma_BC", _bc_rows(live_bc or []), BC_WIDTHS),
            {
                "name": "DE_Search",
                "rows": _search_rows(),
                "freeze": 3,
                "widths": [28, 56, 14, 12, 14, 14, 12, 16],
            },
            {
                "name": "Live_Tweaks",
                "rows": _live_tweak_rows(),
                "freeze": 3,
                "widths": [8, 8, 44, 56, 16, 14, 12, 14, 16, 72],
            },
            {
                "name": "Methodology",
                "rows": _methodology_rows(),
                "freeze": 1,
                "widths": [36, 88],
            },
            {
                "name": "AntiCheat",
                "rows": _anticheat_rows(),
                "freeze": 1,
                "widths": [44, 88],
            },
        ]
    )
    return write_xlsx(path, sheets)
