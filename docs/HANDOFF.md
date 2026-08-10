# Final project handoff

Current share-ready project state. PRs #22 and #23 established the merged baseline;
the canonical-predictor cleanup is protected by the public website contract test.

## What shipped

The repository contains a working NBA game-intelligence system with:

- one frozen logistic-regression win predictor used by Model-only and Agent/Both;
- ridge-regression points, rebounds, and assists projections;
- seven live, date-gated agent tools and one Markdown skill per tool;
- structural snapshots plus query-time cutoff enforcement;
- model-only, agent-only, and agent-plus-model evaluation arms;
- sportsbook-price reconstruction and betting backtests;
- a focused Agent / Compare / System website;
- an inspectable live run context and scoped per-tool gate receipts.

The validated win predictor was trained on 2023-24 and 2024-25, then tested on all
1,322 games of 2025-26. Its holdout accuracy is 66.5%, versus 55.5% for always
picking the home team and 69.0% for the closing line. At the reconstructed prices,
$100 per game loses $2,135 over the season. This is a measured decision-support
system, not a demonstrated profitable betting strategy.

## Clean-clone path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.build_site_data --check
pytest -q

cd ui/web
bun install
bun run lint
bun run build
cd ../..

python -m ui.serve
```

Open `http://localhost:8000`. A live Agent run additionally needs:

```bash
ollama serve
ollama pull gemma4
```

## Supported demo path

1. Open **System** and explain the gate before showing a prediction.
2. Open **Agent**, keep **Both** selected, and run
   `CHI-ORL-2025-12-01` with cutoff `2025-11-30`.
3. Expand the gate receipts and message context.
4. Open **Compare** and distinguish the 80-game paired experiment from the
   1,322-game season result.
5. State the limitation plainly: 66.5% does not beat the 69.0% closing market,
   and the season backtest is negative.

The normal presentation view exposes Agent, Compare, and System. Add
`?details=1` to expose Prompt, Tools, and Data reference tabs.

## Reproduce the evidence

```bash
python -m agent.run --status --source real
python -m eval.three_arms
python -m eval.betting
python -m eval.betting --validate
python -m eval.injury_impact
python -m scripts.gate_snapshot --as-of 2026-01-14
python scripts/skills_doc.py
```

Expected current status:

- 7 of 7 tools live;
- 95 tests passing;
- generated site data current;
- frontend lint and production build passing;
- Model-only and Agent/Both return the same canonical win probability for the
  same matchup and cutoff.

## Intentional scope decisions

- `retrieve_betting_line` was removed from the agent because it exposed the
  evaluation answer. Betting lines remain evaluation-only.
- `retrieve_news` was cut because no source with trustworthy publication
  timestamps was available.
- `predict_best_player` was cut. Player stat projections shipped, but the
  separate ranking tool did not.
- Gate receipts inspect declared serialized date fields. They are visible
  evidence, not the entire leakage guarantee; snapshots and regression tests
  provide the stronger boundary.

## Known limitations

- The paired agent comparison contains 80 games.
- Injury records are transaction dates, not news timestamps.
- Player-to-team injury reconciliation is imperfect after trades.
- Two historical data sources still lack ideal upstream provenance.
- The local agent service has no production authentication or hosting model.
- The LLM can still write a poor explanation even when every supplied fact is
  correctly gated.

## Canonical ownership

- Win model: `models/predict.py` and `models/win_probability.json`
- Player stat model: `models/notebook_model_output.py` plus the stat-line model files
- Agent tools and cutoffs: `agent/tools.py`, `agent/sources.py`
- Agent rules: `skills/`
- Evaluation: `eval/`
- Website API: `ui/serve.py`
- Website: `ui/web/`
- Current narrative: `README.md`, `docs/REPORT.md`, `docs/DEMO.md`
