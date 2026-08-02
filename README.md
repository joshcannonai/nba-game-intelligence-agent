# NBA Game Intelligence Agent

CECS 499 capstone (Summer 2026, UTK). An agentic prediction system for NBA games: pick a matchup, get a structured pregame report with win probability, projected stat lines, matchup context, and a plain-language explanation of what drove the prediction.

**Team:** Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel
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
| `models/` | Regression + XGBoost training, statistical summarization study | Sarvesh |
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

# Full LangChain agent loop -- two backends, pick with --model
cp .env.example .env   # then set ANTHROPIC_API_KEY
python -m agent.run --source real --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24            # Claude, build mode
python -m agent.run --model ollama --source real --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24  # local Gemma 4, no API key

# Leakage guarantees
pytest

# Which tools have data, and where the missing inputs come from
python -m agent.run --status --source real

# Replay the season and score it against baselines
python scripts/build_2026_testset.py          # once, builds the test set
python -m eval.replay --playoffs              # the 85-game held-out set
```

## Evaluation

The team's 2026-07-21 decision: **the 2026 playoffs are the test set**, the regular
season is context. `eval/replay.py` walks the season game by game, sets `as_of` to the
morning before tip-off, asks `predict_win_probability` using only what was knowable
then, and scores the answer three ways — accuracy, log loss, and Brier — against two
baselines.

### The three arms

`eval/three_arms.py` scores three ways of answering the same question, differing by
**exactly one tool** (a test enforces that — the difference *is* the measurement):

- **A** — the model alone, no language model involved
- **B** — the agent alone, reasoning from the retrieval tools
- **C** — the agent plus the model's number

```bash
python -m models.train                            # fit the model, ~7s
python -m eval.three_arms                         # arm A + baselines, ~3s
python -m eval.three_arms --arms abc --sample 40  # all three arms
```

Arm A on **all 1,322 games of 2025-26**, a season the model never trained on:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always-pick-home | 55.5% | 0.6871 | 0.2470 |
| **arm A — logistic regression** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

Trained on 2023-24 and 2024-25. Train/test accuracy gap is **+0.3%**, so it is not
overfit. Split by **season**, not by random shuffle — a random split lets a model learn
from March to predict January, which flatters accuracy by a few points that vanish the
moment anyone checks.

**Honest note on what the model bought us.** The hand-tuned heuristic it replaced
already scored 66.3%, so on raw accuracy the fitted model gains almost nothing. It wins
on calibration (log loss 0.612 vs 0.617, Brier 0.212 vs 0.222) and, more importantly,
on being *checkable*: its split is enforced by tests, its weights are readable, and it
generalises measurably rather than being tuned by hand against the same season it is
scored on. Do not quote it as a big accuracy jump. It is not one.

Arms B and C call a language model once per game (~38s), so the full season in all
three arms is roughly 30 hours. They run on a fixed random sample instead, and every
arm is scored on **the same** games — a paired comparison on 40 beats an unpaired one
on 1,322. The harness prints one standard error next to the headline gap, because at
n=40 that band is about ±8% and wider than the effect we are looking for.

`eval/replay.py` (the older single-arm harness) still exists and still scores
`predict_win_probability` against Vegas across the season.

### Why the betting line lives in its own file

The raw odds source keeps `score_away` / `score_home` in the **same row** as the line.
A retrieval tool reading that row would hand the agent the final score.
`scripts/build_2026_testset.py` splits it into two files that cannot leak into each
other:

- `data/samples/game_logs_2026.csv` — schedule + results. The answer key. Read only by
  the eval harness, *after* a prediction is made. No tool can reach it.
- `data/samples/odds_only.csv` — the market's price, every season 2008-2026.
  **No score columns, ever** (built by `scripts/odds_only.py` from an allowlist).
  The eval harnesses read this. The agent no longer can.

Per the advisor (2026-07-21), the line is an **evaluation baseline, not a model
input** — otherwise the system reads the answer off the market instead of predicting.

That turned out to need more than a rule. Running the live agent on 2026-01-14, it
wrote *"The closing betting line favors the home team, ORL (-5.5 spread)"* straight
into its own key factors — quoting the benchmark back as its reasoning. So
`retrieve_betting_line` was **removed from the agent** in week 6.
`agent.sources.closing_line` remains for scoring, and `tests/test_date_gating.py` now
asserts the tool cannot come back.

Build mode uses Anthropic (personal credits) for fast iteration. Replay / production runs use `--model ollama` -- a local Gemma 4 model (`ollama pull gemma4`) with a known knowledge cutoff so we do not leak future results; Claude's cutoff isn't something we can pin to a date the same way.

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
