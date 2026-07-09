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

## Quick start (agent scaffold, Week 1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# No API key needed: exercises tools + prints a sample report
python -m agent.run --dry-run

# Full LangChain agent loop (needs personal Anthropic credits)
cp .env.example .env   # then set ANTHROPIC_API_KEY
python -m agent.run --matchup LAL-BOS-2026-01-15 --as-of 2026-01-14
```

Build mode uses Anthropic (personal credits) for fast iteration. Replay / production runs will use a local Ollama model with a known knowledge cutoff so we do not leak future results.

Mock matchup lives in `data/mock/`. Tool signatures already take an `as_of_date` so Patrick/Kirtan's date-gated retrieval can drop in later without rewriting the agent.

## Working agreements

- Python 3.11+. Dependencies in `requirements.txt`.
- Feature branches + pull requests. No direct pushes to `main` once code lands.
- Never commit API keys. Copy `.env.example` to `.env` locally (gitignored).
- AI-assisted code is fine per course policy, but you own and can explain every line you merge.
