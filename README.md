# NBA Game Intelligence Agent

CECS 499 capstone (Summer 2026, UTK). An agentic prediction system for NBA games: pick a matchup, get a structured pregame report with win probability, projected stat lines, matchup context, and a plain-language explanation of what drove the prediction.

**Team:** Josh Cannon · Patrick Haley · Sarvvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

## How it works

Two layers do different jobs:

- **Prediction models (classical ML):** linear regression for stat lines, XGBoost for win probability. Fed clean engineered features, exposed as callable tools.
- **Analyst agent (LLM):** decides what data to retrieve for a specific matchup (injuries, head-to-head, fatigue splits), invokes the models, and writes the reasoning.

All data lives in a date-gated store: every query carries an as-of date and only returns records from before it. That lets the evaluation harness replay the 2025-26 games that postdate the local eval model's knowledge cutoff with zero leakage, run data-source ablations, and compare LLM-direct vs. classifier vs. hybrid prediction.

Full plan: see the Proposal Development Plan (linked in the course submission).

## Repo structure

| Folder | What goes here | Owner |
|---|---|---|
| `data/` | Source integrations, cleaning pipeline, feature engineering, date-gated store | Patrick + Kirtan |
| `models/` | Regression + XGBoost training, statistical summarization study | Sarvvesh |
| `agent/` | Tool-use loop, agent actions, model-invocation interface | Josh |
| `eval/` | Replay harness, metrics, ablation runner | Kirtan (with Patrick on store design) |
| `ui/` | Streamlit app | Josh + Kirtan (weeks 6-7) |
| `docs/` | Shared contracts and notes | team |

Roles locked after the 2026-07-07 PDP review with Prof. Sadovnik. Architecture spine: **date-gated retrieval** (every query has an as-of date; no future leakage). Agent builds in parallel on mock / partial data so Josh is not blocked waiting on the full data layer.

Shared tool shapes: see [`docs/tool-contracts.md`](docs/tool-contracts.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# No API key needed: exercises the tools + prints a report
python -m agent.run --dry-run                        # mock fixture
python -m agent.run --dry-run --source real \
    --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24  # real data, date-gated

# Full LangChain agent loop (needs personal Anthropic credits)
cp .env.example .env   # then set ANTHROPIC_API_KEY
python -m agent.run --source real --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24

# Leakage guarantees
pytest
```

Build mode uses Anthropic (personal credits) for fast iteration. Replay / production runs will use a local Ollama model with a known knowledge cutoff so we do not leak future results.

### Two data sources, one contract

`--source mock` reads `data/mock/` — deterministic, no data files, what the tests run on.
`--source real` reads the datasets on `main`, filtered so nothing published after `--as-of` reaches the agent. Same tool signatures either way, which is the point: the data layer can be swapped without touching the agent.

The date gating is real, not decorative. Same game, three as-of dates:

| as-of | players known out | rest | H2H games known |
|---|---|---|---|
| 2024-11-01 | 1 (Porziņģis) | 1d / 1d | 2 |
| 2024-12-01 | 6 | 1d / 1d | 2 |
| 2024-12-24 | 1 (Jaxson Hayes) | 1d / 1d | 2 |

Injury knowledge moves with the as-of date; rest does not (the schedule is published in August — see `docs/tool-contracts.md` for where that line sits, and why the season-aggregate CSVs would leak if used naively).

### Game logs

The datasets on `main` are season aggregates with no schedule or results, so rest / back-to-back / H2H and the eval harness have nothing to stand on. `scripts/fetch_game_logs.py --season 2025` pulls a thin game-by-game table via `nba_api` into `data/samples/`. This is a **sample so shapes match live data**, not the data layer — the real scrape stays with Patrick + Kirtan.

## Working agreements

- Python 3.11+. Dependencies in `requirements.txt`.
- Feature branches + pull requests. No direct pushes to `main` once code lands.
- Never commit API keys. Copy `.env.example` to `.env` locally (gitignored).
- AI-assisted code is fine per course policy, but you own and can explain every line you merge.
