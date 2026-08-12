# Demo runbook

> **Superseded runbook.** This documents the 2026-08-04 prompt and sample results.
> Use the repository README for the current submission demo and evaluator.

Every command below was run end to end on 2026-08-04 and the timings are real.
Two terminals: one for `ollama serve`, one for everything else.

```bash
cd nba-game-intelligence-agent
source .venv/bin/activate
ollama serve &            # only needed for step 5
```

---

## The five-minute version

Run these in order. It builds an argument rather than touring features.

### 1. What is actually built — 2 seconds

```bash
python -m agent.run --status --source real
```

Prints all 7 tools returning real data. **Generated from the code**, not a list
maintained by hand, so the status cannot quietly drift from the implementation.

### 2. The date filter, showing its work — 2 seconds

```bash
python -m scripts.gate_snapshot --as-of 2026-01-14
```

```
samples/game_logs_2026.csv        1,322 kept    719 outcomes cleared
samples/odds_only.csv            23,714 kept    726 dropped
raw/injury_pst_2025_2026.csv      2,272 kept  1,309 dropped
```

This is the architecture the advisor asked for on 2026-07-28: plain Python, no AI,
run *before* the agent. Note the wording — future games keep their row and lose their
result. The agent must be able to see that a game exists without seeing how it ended.

### 3. The predictor against the market — 3 seconds

```bash
python -m eval.three_arms
```

66.5% on 1,322 games it never trained on, against 55.5% for always picking home and
69.0% for the closing line.

### 4. What that is worth in money — 3 seconds

```bash
python -m eval.betting
```

```
                      won      profit
our predictor        66.5%    -$2,135
always the favourite 69.0%    -$4,628
always the home team 55.5%    -$7,350
```

**The line to say out loud:** backing the favourite wins *more games* and loses nearly
*twice as much money*. Everything loses, because the house margin is the bar.

If anyone questions the prices, that is the good question and there is an answer:

```bash
python -m eval.betting --validate
```

19,807 earlier games in the same file carry real quoted prices. Ours reconstruct them
at correlation **0.9959**, average error 2.9 points. The house margin is measured at
3.75%, not assumed.

### 5. The agent itself — 50 seconds

```bash
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```

Runs on the laptop. No API key, no cost, and a knowledge cutoff that predates the
season being predicted. Returns structured JSON: win probability, the factors behind
it, and anything it could not find out.

Worth pointing at: it returns **0.8379**, the predictor's exact number. Before the
skills layer, this specific game was one of the two worst cases in the report — the
agent overrode a confident correct call and got it wrong.

---

## If there is time

### The interface

```bash
streamlit run ui/app.py     # localhost:8501
streamlit run ui/chat.py    # localhost:8502
```

**To put it on a public URL** (still running on this laptop, Gemma included):

```bash
./scripts/share.sh          # prints a https://….trycloudflare.com link
```

Nothing is deployed. Cloudflare forwards a public hostname to localhost, which is
what lets the site keep using the cutoff-pinned local model instead of an API model
that may have the 2025-26 results in its weights. No auth on it, and it dies when the
laptop sleeps, so treat the link as semi-private and short-lived.

Four tabs. The one to show is **Date-gating proof**: the same game at three different
as-of dates, with the injury list changing between them. That is the whole thesis in
one screen.

Tick **"Pre-gate data on disk"** in the sidebar first — the page then reports how many
rows were dropped before anything ran.

### The confound

```bash
python -m eval.injury_impact
```

Three comparisons of the same question. Two come out backwards and one of those is
statistically significant. Comparing each team against itself gives +0.0%. Good if the
conversation turns to methodology.

### The tests

```bash
pytest
```

95 tests, about 100 seconds. The mutation table in the report is the part worth
mentioning: each leakage rule was broken deliberately to confirm the tests caught it.

---

## Answers to have ready

**"Is this profitable?"** No. Everything loses to the house margin. The predictor loses
least.

**"Why is arm A up $993 on the 80-game sample?"** Because 80 games is not enough. The
same predictor loses $2,135 over the full season. It is our own demonstration of the
trap the report is about.

**"How do you know the AI isn't just remembering?"** Gemma 4's cutoff is around January
2025, verified by asking it about 2024, 2025 and 2026 results. Every test game postdates
it. The 2023-24 and 2024-25 seasons are demos of the mechanism, not valid test games.

**"Did you write the model that was Sarvesh's?"** The interface was ours either way and
nothing downstream could be scored without something behind it. `models/README.md` is
written as a handoff, and swapping his in is one file.

**"Where did the odds come from?"** We don't know, and that is a real gap. Kirtan
supplied the file and the URL was never recorded. The whole betting benchmark rests on
it.

---

## If something breaks

| Symptom | Cause |
|---|---|
| `ConnectError: Connection refused` on step 5 | `ollama serve` is not running |
| Streamlit shows stale output | No auto-reload here. Restart the server after any edit. |
| `ModuleNotFoundError` | `source .venv/bin/activate` |
| Agent returns `p=None` | The model failed. The harness records a skip rather than guessing 0.5 — that guard is deliberate. |
