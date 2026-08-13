# Verified actual-UI evaluation

The professor packet for CECS 499 is Models A, B, and C only. D and E
(market-aware follow-ons) are deferred and are **not** required submission
content.

## Professor packet (full season, A/B/C)

- `NBA-Actual-UI-Agent-Evaluation.xlsx` — hand this workbook to Prof. Sadovnik
- `verified-full-season-abc-summary.csv` — Git-diffable headline
- `verified-full-season-abc-results.csv` — Git-diffable game table
  (A/B/C × 1,322 games; compact columns, no B/C narratives)
- `manifest.json` — SHA-256, contract, caveats

Rebuild after the local B/C checkpoint exists:

```bash
PYTHONPATH=vendor python -m eval.build_abc_packet
```

### Full 2025-26 season (1,322 games, previous-day cutoff)

| Arm | Path | Correct | Accuracy | $100 P&L |
|---|---|---|---|---|
| A | `POST /api/predict` | 871/1322 | 65.9% | −$2,720.60 |
| B | `POST /api/run` SSE, Gemma 4, five retrieval tools | 813/1322 | 61.5% | −$5,229.85 |
| C | same as B + `predict_win_probability` | 860/1322 | 65.1% | −$3,175.21 |

Hypothesis was that C beats A and B. C beat B. C did not beat A. All three
lose money at $100/game versus reconstructed vig (A least bad). Prices are
reconstructed from the closing spread (σ=14, 3.75% hold), not quoted
moneylines and not tickets.

A is not in the B/C jsonl. Game-level A rows were joined from
`full-season-mass-eval.csv` onto the same 1,322 `game_id`s, then settled with
the same B/C rule (`stake × (decimal − 1)` if correct, else −$100, 2-decimal).
The mass-eval CSV stores A P&L to 4 decimals (−$2,720.97); the 37-cent gap is
rounding only.

The old 10-game classroom workbook
(`NBA-Actual-UI-Agent-Evaluation-Shared-10-Games.xlsx`) is not part of this
packet. Do not submit it.

## What the arms are

- Model A executed through `POST /api/predict`
- Models B and C executed through the website's `POST /api/run` SSE path
- Model B used `gemma4:latest` and the five required retrieval calls
- Model C used the same language model and retrieval calls, plus Model A's
  predictor as an additional tool
- A/B/C do not see Vegas
- every required B/C gate receipt passed (0 failures)

Caveat: 5 of 1,322 Model B games also called `predict_win_probability` (B is
specified as retrieval-only). They still include the five required retrieval
calls. They are not dropped from the headline.

## Full-season mass-eval (deferred D/E)

CECS 499 is Models A, B, and C. D and E live in `eval/policies.py` /
`eval/mass_eval.py` for later self-improvement work, not this submission.
`NBA-Full-Season-Mass-Eval-Betting.xlsx` is a D/E artifact — do not submit it
as the class packet. Do not run `python -m eval.gemma_de` for this packet.

The 2025-26 odds file has spreads and no quoted moneylines. Prices are
reconstructed from the closing spread with a 3.75% hold; validate with
`python -m eval.betting --validate`. Polymarket historical closes are not in
the repository yet.

The agent narratives may cite records and ratings from the prior completed
season because that is the date-gated historical context available to the
agent. Those figures are not current-season final records and should not be
read as post-cutoff information.

To reproduce the underlying actual-UI result CSV, build the web client and run
the bridge in one terminal, then run the evaluator in a second terminal:

```bash
cd ui/web
npm install
npm run build
cd ../..
python -m ui.serve
```

```bash
python -m eval.ui_agent_eval --arms A,B,C \
  --checkpoint eval/results_actual_ui_10.jsonl \
  --out eval/results_actual_ui_10.csv
```

The evaluator requires the local `gemma4:latest` Ollama model for Models B and
C. Evaluation checkpoints and raw result CSVs under `eval/` are gitignored
because they are machine-specific runtime outputs. The committed CSVs in this
folder are the bounded, professor-facing tables exported from the reviewed
packet.
