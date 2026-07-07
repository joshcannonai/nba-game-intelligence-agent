# NBA Game Intelligence Agent

CECS 499 capstone (Summer 2026, UTK). An agentic prediction system for NBA games: pick a matchup, get a structured pregame report with win probability, projected stat lines, matchup context, and a plain-language explanation of what drove the prediction.

**Team:** Josh Cannon · Patrick Haley · Sarvvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

## How it works

Two layers do different jobs:

- **Prediction models (classical ML):** linear regression for stat lines, XGBoost for win probability. Fed clean engineered features, exposed as callable tools.
- **Analyst agent (LLM):** decides what data to retrieve for a specific matchup (injuries, head-to-head, fatigue splits), invokes the models, and writes the reasoning.

All data lives in a date-gated store: every query carries an as-of date and only returns records from before it. That lets the evaluation harness replay the completed 2025-26 season game by game with zero leakage, run data-source ablations, and compare LLM-direct vs. classifier vs. hybrid prediction.

Full plan: see the Proposal Development Plan (linked in the course submission).

## Repo structure

| Folder | What goes here | Owner |
|---|---|---|
| `data/` | Source integrations, cleaning pipeline, feature engineering, date-gated store | Patrick |
| `models/` | Regression + XGBoost training, statistical summarization study | Sarvvesh |
| `agent/` | Tool-use loop, agent actions, model-invocation interface | Josh |
| `eval/` | Replay harness, metrics, ablation runner | Kirtan |
| `ui/` | Streamlit app | Josh + Kirtan |

## Working agreements

- Python 3.11+. Dependencies in `requirements.txt` (coming with the first code).
- Feature branches + pull requests. No direct pushes to `main` once code lands.
- Never commit API keys. Copy `.env.example` to `.env` locally (gitignored).
- AI-assisted code is fine per course policy, but you own and can explain every line you merge.
