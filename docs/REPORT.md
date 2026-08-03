# NBA Game Intelligence Agent — Final Report

**CECS 499 · Senior Transdisciplinary Capstone · Summer 2026 · University of Tennessee, Knoxville**

**Team:** Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

**Repository:** `github.com/joshcannonai/nba-game-intelligence-agent`

---

> **DRAFT STATUS — v1, 2026-08-03.** Sections 1–7, 9, 10, 11 and Appendix A are written
> against verified, reproducible output from the repository. Sections 8 and 12 are
> **shells awaiting their owners** (Sarvesh's models, Patrick's data pipeline, Kirtan's
> gating function). Every number in this draft was regenerated from a clean checkout on
> 2026-08-03; the commands that produce each one are in Appendix A.

---

## Contents

1. [Abstract](#1-abstract)
2. [The problem, and why it is harder than it looks](#2-the-problem-and-why-it-is-harder-than-it-looks)
3. [System architecture](#3-system-architecture)
4. [Date gating — the core mechanism](#4-date-gating--the-core-mechanism)
5. [The agent](#5-the-agent)
6. [The win-probability model](#6-the-win-probability-model)
7. [The experiment: three arms](#7-the-experiment-three-arms)
8. [The models teammates own — *shell*](#8-the-models-teammates-own--shell)
9. [Results](#9-results)
10. [Discussion: why the negative result is the interesting one](#10-discussion-why-the-negative-result-is-the-interesting-one)
11. [Limitations and threats to validity](#11-limitations-and-threats-to-validity)
12. [Team contributions — *shell*](#12-team-contributions--shell)
13. [AI use disclosure](#13-ai-use-disclosure)
    - [Appendix A — Reproducing every number](#appendix-a--reproducing-every-number-in-this-report)
    - [Appendix B — Repository map](#appendix-b--repository-map)

---

## 1. Abstract

We built a system that predicts the outcome of NBA games and explains its reasoning, and
then we ran an experiment to find out whether the explaining part helps.

The system has two halves that answer the same question by different means. A **logistic
regression** reads eight engineered features and returns a win probability. An **LLM
agent** calls seven retrieval tools, reasons over what it finds, and writes a pregame
report. Both are constrained by the same rule: every query carries an *as-of date*, and
nothing published after that date may reach either one. That constraint is what makes it
possible to test on the 2025-26 season — a season that has already happened, and that a
language model might simply remember.

On all 1,322 games of 2025-26, held out from training, the model scores **66.5% accuracy**
against **55.5%** for always picking the home team and **69.0%** for the Vegas closing
line.

Our stated hypothesis was that an agent given the model's number would beat the model
alone. **It did not.** On paired samples the agent-plus-model arm scored *worse* than the
model by itself, and the mechanism is specific: when the agent overruled the number it was
handed, it was wrong **15 times out of 19** (pooled over two 40-game samples, one-sided
sign test p ≈ 0.010; two-sided p ≈ 0.019). The explanation layer, as built, costs accuracy
rather than adding it. That negative result is the most interesting thing we found and it
is what this report leads with.

---

## 2. The problem, and why it is harder than it looks

Predicting NBA games is a well-worn problem with an obvious benchmark: the betting market
prices every game, and the closing line is very hard to beat. What makes the problem
interesting for a capstone is not the prediction. It is the **evaluation**.

To know whether a system is any good, you have to test it on games it has never seen. For
a classical model that is routine — hold out a season. For a system built on a large
language model it is a genuine methodological hazard, because the model has read the
internet. Ask a modern LLM who won a game in December 2025 and it may simply know. Any
accuracy number produced that way measures memory, not prediction.

This is the same class of problem our advisor described in the OpenAI `exploit-gym`
incident: a system optimising for a goal will take whatever route reaches it, and the
route that reaches it fastest is often the one we would call cheating. As he put it, the
model *"doesn't even know what cheating is."* You cannot fix that by asking nicely in a
prompt. You have to make the shortcut structurally unavailable.

So the project's real subject is **leakage control**, and the prediction task is the
vehicle for demonstrating it.

### 2.1 Three distinct leaks

They are usually collapsed into one word — "leakage" — and they need different defences.

| # | Leak | Concretely | Defence |
|---|---|---|---|
| 1 | **Data leakage** | A retrieval tool returns an injury report filed *after* the game | Date gating (§4) |
| 2 | **Model-knowledge leakage** | The LLM remembers the result from training | Cutoff-pinned local model (§4.4) |
| 3 | **Benchmark leakage** | The system reads the answer off the thing we grade it against | Tool removal (§5.3) |

Leak 3 is the one we did not anticipate, caught in the act, and had to fix by design
change rather than by rule. It is covered in §5.3.

---

## 3. System architecture

```
                    ┌───────────────────────────────────────┐
   user picks       │  scripts/gate_snapshot.py --as-of D   │   plain Python.
   a game + a  ───▶ │  copies data → data/snapshots/D/      │   no AI involved.
   date D           │  future results physically absent     │
                    └──────────────────┬────────────────────┘
                                       │
                    ┌──────────────────▼────────────────────┐
                    │  agent/sources.py                     │   second gate:
                    │  every read filtered to ≤ D           │   per-tool precision
                    └───────┬───────────────────┬───────────┘
                            │                   │
              ┌─────────────▼──────┐   ┌────────▼───────────────┐
              │  7 agent tools     │   │  models/features.py    │
              │  agent/tools.py    │   │  8 features, as-of D   │
              └─────────┬──────────┘   └────────┬───────────────┘
                        │                       │
              ┌─────────▼──────────┐   ┌────────▼───────────────┐
              │  LLM agent loop    │   │  logistic regression   │
              │  agent/run.py      │   │  models/predict.py     │
              │  (arms B, C)       │   │  (arm A)               │
              └─────────┬──────────┘   └────────┬───────────────┘
                        └───────────┬───────────┘
                                    │
                    ┌───────────────▼───────────────────────┐
                    │  eval/three_arms.py                   │  reads results
                    │  accuracy · log loss · Brier          │  ONLY to score,
                    │  vs always-home, vs Vegas             │  after the fact
                    └───────────────────────────────────────┘
```

Two things about this diagram matter more than the boxes.

**The gate runs twice, deliberately.** The snapshot removes what *nobody* may see. The
query-time filter decides what *each tool* may see. They are not redundant — §4.3 explains
why collapsing them into one would be worse.

**The model and the agent share their data access.** `models/features.py` imports the same
gated accessors from `agent/sources.py` that the agent's tools use. This is not a
convenience; it is what makes the three-arm comparison meaningful. If the model could see
anything the agent could not, the experiment would be measuring our plumbing instead of
our ideas. `tests/test_model_contract.py` asserts the two implementations of rolling form
agree game for game, so if they ever drift apart a test fails rather than the model
quietly training on a different world than the agent inhabits.

---

## 4. Date gating — the core mechanism

### 4.1 The rule

Every retrieval carries an `as_of_date`. Only records published on or before that date may
be returned. Anything that cannot be computed comes back as `null` with a stated reason,
never as zero. That last clause matters more than it sounds: an unknown injury list is not
"nobody is hurt," and a system that silently substitutes zero for unknown will look
confident and be wrong.

### 4.2 Gate 1 — the on-disk snapshot

`scripts/gate_snapshot.py` is a plain Python script. No model runs. It reads the data
directory and writes a filtered copy:

```
$ python -m scripts.gate_snapshot --as-of 2026-01-14
Snapshot as of 2026-01-14 -> data/snapshots/2026-01-14

  samples/game_logs_2024.csv                       1,230 kept
  samples/game_logs_2025.csv                       1,225 kept
  samples/game_logs_2026.csv                       1,322 kept      719 outcomes cleared
  samples/odds_only.csv                           23,714 kept      726 dropped
  raw/injury_data_2016_2025/injury_data.csv       16,873 kept
  raw/injury_pst_2025_2026/injury_data.csv         2,272 kept    1,309 dropped
  raw/nba_stats_1947_present/Team Summaries.csv    1,876 kept       31 dropped
  raw/nba_stats_1947_present/Player Per Game.csv  32,606 kept      733 dropped
```

Point the agent at that directory (`NBA_SNAPSHOT_DIR=...`) and the future is not merely
filtered — it is **not on disk**.

One design decision inside this is worth calling out, because the obvious implementation
is wrong. For future games the script does **not** drop the row. It keeps the row and
erases three columns (`home_pts`, `away_pts`, `winner`). The reason: the agent is being
asked to preview a game that has not been played, so it must be able to see that the game
*exists*. What it must not see is how it *ended*. Dropping the row would hide the question
along with the answer.

This gate exists because our advisor asked for it directly on 2026-07-28:

> "The first thing that happens before the agent even runs is that you run [a script] that
> copies a part of the data into a folder that has only [data] up to a certain date. And
> then you run your agent, and your agent can only look at *it*... you're not relying on
> the LLM to gate its own data. You pre-gate it."

### 4.3 Gate 2 — query-time filtering

`agent/sources.py` filters every read against the as-of date independently.

The two gates are complementary rather than belt-and-braces. A snapshot can only be as
strict as its *loosest legitimate reader*. `team_form_as_of` needs games strictly **before**
the as-of date; `schedule_context` needs games **through** it. A single on-disk cut cannot
satisfy both without starving one of them. So the snapshot removes what nobody may see,
and the query-time filter draws the finer per-tool line.

Not every date-sensitive value is a leak, and conflating them would cripple the system.
The NBA publishes its full schedule in August, so "BOS plays on the 23rd and the 25th" is
knowable on any as-of date in the season — rest and back-to-back status are therefore
legitimate features. The *outcome* of the game on the 23rd is not. `schedule_context` makes
that distinction explicitly, and `models/features.py` mirrors it.

### 4.4 Gate 3 — pinning the language model's knowledge

Gating the data does nothing about what the model already knows. A 2025-26 game is inside
the training window of most current commercial models.

So the replay runs on **Gemma 4 via Ollama**, locally, with a knowledge cutoff of roughly
January 2025 — verified behaviourally (it knows the 2024 Finals; it does not know 2025 or
2026). Every game in the 2025-26 test window postdates that cutoff, so the model cannot be
remembering.

This has a consequence the report must state plainly: **the 2023-24 and 2024-25 seasons are
demonstrations of the mechanism, not valid evaluation games for the LLM arms**, because
they predate the cutoff. They remain perfectly valid for training the logistic regression,
which has no world knowledge to leak. A faster commercial model (Claude) is used during
development for iteration speed, and never for a scored replay.

### 4.5 Proving the gate works

Tests that cannot fail prove nothing. Rather than assert that a filter was called, we broke
each rule on purpose and confirmed that tests caught it:

| Mutation introduced | Tests that failed |
|---|---|
| Feature accumulators advanced *before* the row was emitted | 3 |
| Form window drifted out of sync with the agent's accessor | 1 |
| Test season added to the training seasons | 3 |

The first mutation is the important one. The natural way to compute a team's season win
percentage — group the season, take the mean — silently includes the game being predicted.
That yields a very accurate and completely worthless model. `models/features.py` advances
every accumulator *after* emitting the row, never before.

The suite is **64 tests**, run from a clean checkout on 2026-08-03.

---

## 5. The agent

### 5.1 The tool interface

The agent's entire world is seven functions. It cannot query a database, browse the web, or
invent a number — it can only call these, and every retrieval tool takes an as-of date.

| Tool | Status | What it returns |
|---|---|---|
| `retrieve_matchup_context` | ✅ real data | Team ratings, rest, injuries, head-to-head, as of a date |
| `retrieve_player_splits` | ✅ real data | Season averages, optional back-to-back fatigue split |
| `retrieve_team_form` | ✅ real data | Rolling 10-game record and point differential |
| `retrieve_injuries` | ✅ real data | Who was known to be out that morning |
| `predict_win_probability` | ✅ real model | Logistic regression output (withheld in arm B) |
| `retrieve_schedule` | ⏳ awaiting input | Blocked on a committed forward-looking schedule table |
| `predict_stat_line` | ⏳ awaiting input | Points/rebounds/assists regression — not started |

Running `python -m agent.run --status --source real` prints this live, which means the
project's own blocking list is generated from the code rather than maintained by hand.

**Placeholders do not lie.** A tool whose input does not exist returns

```json
{ "status": "awaiting_input", "needs_from": "...", "needs": "..." }
```

and the agent is instructed to report the gap rather than fill it. This is a deliberate
design stance: the failure mode we were most worried about is a system that produces a
confident, complete-looking report with an invented number inside it.

### 5.2 From ten tools to seven

The original design had ten. Three were cut, and a scope cut nobody can explain later just
looks like abandoned work, so:

- **`retrieve_news`** — no source with reliable publication timestamps was ever found.
  Highest effort of the ten, least measurable contribution. Cut on merit.
- **`predict_best_player`** — depended entirely on `predict_stat_line`, which never
  started. A placeholder behind a placeholder.
- **`retrieve_betting_line`** — not a scope cut. A leak. See below.

### 5.3 The tool we had to take away

This is the most instructive incident in the project.

The agent had a `retrieve_betting_line` tool. Its docstring said, in capital letters, that
the line was context only and must not drive the prediction. Running the live agent on
2026-01-14, it wrote this into its own key factors:

> *"The closing betting line favors the home team, ORL (-5.5 spread)"*

The closing line is the benchmark we grade ourselves **against**. An agent that reads the
market and repeats it scores beautifully and has predicted nothing — and it would have
scored beautifully in the exact run we were about to demonstrate.

The instruction was not violated in any dramatic way. The agent was asked for the factors
behind its reasoning and it truthfully reported one. The problem is that the tool made the
shortcut *available*, and an optimiser takes available shortcuts. This is precisely the
alignment failure our advisor described: the system had no concept that consulting the
answer key was off-limits, only a goal to produce a good prediction.

**Telling a model not to peek is a request. Removing the tool is a guarantee.**

So `retrieve_betting_line` was removed from the agent's tool set. `agent.sources.closing_line`
still exists and `eval/` still calls it directly, so the Vegas baseline is untouched — the
line is available to the *scorer* and unavailable to the *predictor*. A test in
`tests/test_date_gating.py` asserts the tool cannot reappear in the agent's tool list.

### 5.4 Keeping the answer key physically separate

The raw odds source stores `score_away` and `score_home` in the **same row** as the betting
line. A retrieval tool reading that row to get the spread would hand the agent the final
score. This is the single most likely way this project could have leaked.

`scripts/odds_only.py` derives the odds sample through a column **allowlist**, so score
columns cannot reach it even if the source schema changes. The result is two files that
cannot contaminate each other:

- `data/samples/game_logs_2026.csv` — schedule and results. The answer key. Read only by
  the eval harness, only after a prediction has been made.
- `data/samples/odds_only.csv` — the market's price. No score columns, ever.

**Verifying the baseline is what we think it is.** The entire Vegas comparison rests on an
assumption: that our odds file contains *closing* lines rather than *opening* lines. Those
differ, and the closing line is the stronger benchmark. `eval/crosscheck_odds.py` tests the
assumption against an independent source that labels the two explicitly, and finds **9 of
10 sampled games are closer to closing**. The baseline is what we claim it is.

---

## 6. The win-probability model

### 6.1 What it is

A **logistic regression** — deliberately, not because it was easiest. It has three
properties that a tree ensemble would not have given us: its weights are human-readable in
a pull request, it serialises to a few hundred bytes of named numbers rather than a pickle,
and it loads without `sklearn`, so the agent, the eval harness and the UI all run without
the training dependency installed.

**Input:** home team, away team, as-of date.
**Output:** `home_win_prob` as a float 0–1, plus provenance.

Eight features, all computed strictly from games *before* the one being predicted:

| Feature | Standardised weight |
|---|---|
| `win_pct_diff` | +0.3957 |
| `form_margin_diff` | +0.3780 |
| `away_games_played` | −0.2731 |
| `injury_weight_diff` | −0.2457 |
| `home_games_played` | +0.1791 |
| `home_back_to_back` | −0.1466 |
| `away_back_to_back` | +0.1211 |
| `rest_diff` | +0.0031 |
| *(intercept)* | +0.2022 |

Weights are standardised, so they are comparable to one another. The signs are all
physically sensible: a better record helps, being on a back-to-back hurts, losing players
to injury hurts, and playing an opponent who is on a back-to-back helps.

### 6.2 The split

**Trained on 2023-24 and 2024-25. Tested on 2025-26.** Split by **season**, not by random
shuffle. A random shuffle lets the model learn from March in order to predict January,
which inflates accuracy by a few points that vanish the moment anyone checks. Two constants
in `models/train.py` — `TRAIN_SEASONS = (2024, 2025)` and `TEST_SEASON = 2026` — encode
this, and three tests fail if the test season is added to the training set.

### 6.3 Results

On **all 1,322 games** of 2025-26:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **our model (arm A)** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

Train accuracy is 66.8% against test accuracy 66.5% — a **generalisation gap of +0.3%**.
The model is not overfit. We sit 2.5 points below the market, which is a respectable place
to be standing given that the market has injury beat reporters and real money and we have
a laptop.

Three metrics rather than one, because accuracy alone is close to meaningless on NBA games:
**log loss** punishes confident wrong answers, and **Brier** is mean squared error on the
probability. A system can be accurate and badly calibrated, and for a system meant to
express uncertainty that distinction matters.

### 6.4 Honest note on what the fitted model bought us

The hand-tuned heuristic this replaced already scored 66.3%. **On raw accuracy, the fitted
model gains almost nothing.** It wins on calibration (log loss 0.612 vs 0.617, Brier 0.212
vs 0.222) and, more importantly, on being *checkable*: its split is enforced by tests, its
weights are readable, and it generalises measurably rather than being hand-tuned against
the same season it is scored on. It should not be quoted as a large accuracy jump, because
it is not one.

---

## 7. The experiment: three arms

### 7.1 Design

Three ways of answering the same question, differing by **exactly one tool**:

| Arm | What it is | Has `predict_win_probability`? | LLM? |
|---|---|---|---|
| **A** | Model only | — (*is* the model) | No |
| **B** | Agent only | **No** | Yes |
| **C** | Agent + model | **Yes** | Yes |

Arms B and C are the same agent, the same prompt, the same data, the same gate. The only
difference is whether `predict_win_probability` is in the tool list — controlled by a
single `include_model` flag in `agent/tools.py`, with a test asserting the two tool lists
differ by that one entry and nothing else. **The difference *is* the measurement.**

**Hypothesis, stated before the run: C beats both A and B.** An agent that can explain
itself should cost nothing in accuracy, and might gain from context the model cannot see.

### 7.2 Why the LLM arms run on a sample

Arms B and C call a language model once per game, roughly 38 seconds each locally. All
1,322 games across all three arms is about **30 hours**. So B and C run on a fixed random
sample of 40, and — this is the part that matters — **every arm is scored on that same
sample**, not on its own convenient subset.

A paired comparison on 40 games beats an unpaired one on 1,322 here, because the arms
differ by exactly one variable. The harness prints one standard error next to the headline
gap, and at n=40 that band is roughly **±8%**, which is wider than the effect we were
looking for. The harness refuses a full-season LLM run rather than quietly starting a
30-hour job.

### 7.3 Independent scoring

Per our advisor's instruction on 2026-07-28, the LLM does **not** grade itself. `eval/three_arms.py`
is a plain Python script: it collects each arm's probability, then compares against ground
truth read from a file no tool can reach. An unparseable LLM answer is recorded as a skip,
never coerced to 0.5 — coercing would quietly drag an arm toward the baseline and flatter
it.

---

## 8. The models teammates own — SHELL

> **Sarvesh** — this section is yours. From the 2026-07-28 review, it needs to answer the
> advisor's standing question directly: *for each model, what is the input and what is the
> output?* Suggested structure:
>
> - The stat-line regression (points / rebounds / assists): input features, output shape,
>   train/test split, results vs a sensible baseline.
> - The XGBoost win classifier: same. **The 07-28 review flagged that the classifier setup
>   may be misconfigured** — resolving that here, one way or the other, is the highest-value
>   thing in this section.
> - The comparison of linear regression vs XGBoost for the stat lines, and which won.
> - Apples-to-apples: the advisor asked that your win predictions and the agent's run on
>   the *same set of games*. §9 has our 1,322-game set; state which games yours covers.
>
> **Note the dependency:** `predict_stat_line` in `agent/tools.py` is written and waiting.
> Drop the regression behind that signature and the agent picks it up with no other change.
> See `models/README.md`.

> **Patrick** — the data pipeline section is yours: sources, why `basketball_reference_web_scraper`
> and the ESPN API rather than `nba_api` (the Codespaces/Azure IP block), the cleaning
> pipeline, and the rolling-5 / rolling-10 engineered features that feed Sarvesh's models.

> **Kirtan** — the gating function you wrote, and how it relates to `scripts/gate_snapshot.py`;
> plus the odds cross-check (§5.4) which is your result.

---

## 9. Results

### 9.1 Full season, arm A (n = 1,322)

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **arm A — logistic regression** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

### 9.2 All three arms, paired samples (n = 40 each)

**Sample 1 (seed 0):**

| arm | accuracy | log loss | Brier |
|---|---|---|---|
| **A — model only** | **75.0%** | **0.5782** | **0.1965** |
| B — agent only | 57.5% | 0.6577 | 0.2323 |
| C — agent + model | 55.0% | 0.6754 | 0.2409 |
| Vegas | 57.5% | 0.6488 | 0.2278 |
| always-home | 50.0% | 0.6982 | 0.2525 |

**Sample 2 (seed 1):**

| arm | accuracy | log loss | Brier |
|---|---|---|---|
| **A — model only** | **70.0%** | **0.5567** | **0.1871** |
| B — agent only | 62.5% | 0.6394 | 0.2242 |
| C — agent + model | 62.5% | 0.6723 | 0.2352 |
| Vegas | 77.5% | 0.4683 | 0.1514 |
| always-home | 62.5% | 0.6731 | 0.2400 |

**Do not quote arm A's 75% as our accuracy.** The season-long figure is 66.5%; these 40
games happened to suit it. The two samples also show how noisy n=40 is — Vegas scored
57.5% on the first and 77.5% on the second, on the same season. That volatility is the
reason the headline finding below is a *paired* one.

### 9.3 The paired finding

Comparing headline accuracies asks whether a particular 40 games flattered one arm.
The paired question is immune to that: **on the games where the agent overruled the model
it was handed, how often did overruling help?**

| sample | agreed | overrides | model right | agent right | one-sided p |
|---|---|---|---|---|---|
| seed 0 | 28 | 12 | 10 | 2 | 0.019 |
| seed 1 | 33 | 7 | 5 | 2 | 0.227 |
| **pooled** | **61** | **19** | **15** | **4** | **0.0096** |

**When the agent overruled the model, it was wrong 15 times out of 19.**

Two caveats stated up front, because both are real:

1. **Seed 1 alone is not significant.** Seven overrides and a 5–2 split is ordinary luck.
   What replicates is the *direction* — the agent's overrides succeed 17% and 29% of the
   time, both far below the 50% a coin would give. Pooled across 19 overrides it holds.
2. **The p-values above are one-sided.** The two-sided pooled value is **p ≈ 0.019**, which
   is still significant at α = 0.05. Since our pre-stated hypothesis (C > A) pointed the
   *other* way, the two-sided figure is the more defensible one to quote, and the
   conclusion survives either test.

It was not overruling at the margins, either. The largest reversals took a confident
correct call and inverted it:

| game | model | agent | actual |
|---|---|---|---|
| CHI-ORL-2025-12-01 | 0.815 | 0.249 | home won |
| IND-PHI-2026-01-19 | 0.741 | 0.242 | home won |

---

## 10. Discussion: why the negative result is the interesting one

We predicted C > A. We got C < B < A. The explanation layer, as built, is **costing**
accuracy.

The paired analysis tells us the failure is not random noise around a good estimate. The
agent is systematically talking itself out of the model's confident, correct calls. Our
leading hypothesis is that it **over-weights the injury list** available through
`retrieve_injuries`: seeing four names on a list reads as decisive to a language model,
whereas the fitted model has learned from two seasons roughly how much a given injury load
is actually worth. The model has a calibrated prior about injuries; the agent has a
narrative one.

This mirrors a failure mode the earlier hand-tuned heuristic had, and it points at a
concrete redesign rather than a vague conclusion: the agent should probably be permitted to
*annotate* the model's number rather than *replace* it — a shift from the agent as
predictor to the agent as explainer, with the model retaining the final say on the
probability.

We want to be careful not to over-claim from this. What we have shown is that **this**
agent, with **these** seven tools, on **these** 79 paired games, degrades a good estimate.
We have not shown that LLM agents cannot improve on classical models in general. The honest
scope of the claim is the narrow one.

It is worth saying plainly that this is a more useful result than a confirmed hypothesis
would have been. A confirmed hypothesis would have told us our architecture was fine. This
tells us where it is broken, and it came out of an experimental design — paired arms
differing by one tool, independent scoring, a pre-stated hypothesis — built specifically so
that a negative answer would be legible instead of ambiguous.

---

## 11. Limitations and threats to validity

Stated deliberately, because the ones we can name are less dangerous than the ones we cannot.

1. **n = 40 per LLM sample.** Roughly a ±8% band on accuracy. The headline paired finding
   pools to 19 overrides, which is enough for a sign test and not much more.
2. **`predict_stat_line` was never built.** "Projected stat lines" was a stated deliverable
   in the PDP. The tool signature exists and returns `awaiting_input`; the regression behind
   it does not. This is a scope shortfall, not a hidden one — the status board reports it.
3. **Injury data are transaction dates, not news timestamps.** The log records when a player
   was placed on or activated from the injured list, not the moment the news broke. An
   as-of query on the morning of a game can therefore see a same-day placement that a
   real-time user might not have had. This is a residual leak of unknown but probably small
   size, and we would rather name it than let it be found.
4. **Injury importance is a prior-season minutes/points proxy** and treats every listed
   player as fully out. It over-penalises. Players with no prior season carry `None`, not
   `0.0` — a rookie is unknown, not worthless.
5. **`home_games_played` and `away_games_played` carry meaningful weight**, which is mildly
   suspicious: the two are near-identical for any given game, so the model may be fitting a
   schedule artifact rather than a basketball fact. Worth an ablation we did not run.
6. **No opponent-adjusted strength of schedule.** A 5–0 run against bad teams counts the
   same as 5–0 against good ones. Probably the largest single available improvement.
7. **The model-knowledge gate only holds for 2025-26.** Earlier seasons predate Gemma 4's
   cutoff and are demonstrations of the mechanism, not valid LLM evaluation games.
8. **The UI's report tab is not agentic.** It runs the deterministic no-LLM path.
   `ui/chat.py` can be agentic if the backend is switched in the sidebar. We say so in the
   interface rather than letting a demo imply otherwise.
9. **Vegas is an in-sample baseline in one respect:** the spread-to-probability conversion
   uses a residual σ fitted on the same 1,322 games. That makes the baseline slightly
   *stronger* than it deserves — a conservative bar, but not a neutral one.

---

## 12. Team contributions — SHELL

> Each member completes their own row. Accuracy here matters: the advisor will ask.

| Member | Lane | Delivered |
|---|---|---|
| **Josh Cannon** | Agent | The 7-tool interface (`agent/tools.py`), the agent loop (`agent/run.py`), date-gated sources (`agent/sources.py`), the snapshot gate (`scripts/gate_snapshot.py`), the win-probability model (`models/`), the replay and three-arm harnesses (`eval/`), 64 tests, both Streamlit UIs (`ui/`). |
| **Patrick Haley** | Data | *(Patrick to complete — collection pipeline, cleaning, rolling-5/10 engineered player and team features, PRs #12, #14, #15.)* |
| **Sarvesh Vinod Kumar** | Models | *(Sarvesh to complete — linear regression and XGBoost for stat lines, XGBoost win classifier, accuracy evaluation notebook.)* |
| **Kirtan Patel** | Data / gating | *(Kirtan to complete — the date-gating function, candidate dataset survey, odds cross-check.)* |

---

## 13. AI use disclosure

Per course policy, and stated in full.

This project used AI coding assistance (Claude) extensively throughout — for
implementation, refactoring, test authoring, and documentation drafting, including drafting
portions of this report. That assistance is disclosed rather than concealed because the
course requires it and because the alternative would misrepresent how the work was done.

Two commitments constrain what that assistance was allowed to produce:

1. **Every line merged is understood and defensible by its author.** The architecture
   decisions in this report — the two-layer gate, the tool removal in §5.3, the paired
   experimental design, the choice of logistic regression over a tree ensemble — were made
   by us, and each is argued from a reason rather than a convention.
2. **No result in this report was generated by an LLM judging its own output.** Every
   number was produced by plain Python scoring against ground truth in a file no agent tool
   can reach, and every number in this draft was regenerated from a clean checkout on
   2026-08-03.

The project's central finding is itself a caution about uncritical AI use: an LLM agent,
given a tool it was told not to lean on, leaned on it — and given a good estimate, made it
worse. We treat that as evidence for verification over trust.

---

## Appendix A — Reproducing every number in this report

From a clean clone, on Python 3.11+:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                    # 64 tests — the leakage guarantees
python -m models.train                    # §6.1 weights, §6.3 accuracy, the +0.3% gap
python -m eval.three_arms                 # §9.1 — arm A + both baselines, 1,322 games
python eval/crosscheck_odds.py            # §5.4 — 9 of 10 closer to closing
python -m scripts.gate_snapshot --as-of 2026-01-14   # §4.2 — the snapshot manifest
python -m agent.run --status --source real           # §5.1 — the live tool board
```

The paired-arm results in §9.2 and §9.3 are committed as
`eval/results_three_arms_sample40.csv` and `..._seed1.csv`. To regenerate (requires
`ollama serve` and `ollama pull gemma4`; ~50 minutes each):

```bash
python -m eval.three_arms --arms abc --sample 40 --model ollama
python -m eval.three_arms --arms abc --sample 40 --seed 1 --model ollama
```

The user interface:

```bash
streamlit run ui/app.py     # pregame report, tool inventory, gating proof, build status
streamlit run ui/chat.py    # conversational view over the same tools and the same gate
```

## Appendix B — Repository map

| Path | Contents |
|---|---|
| `agent/` | Tool definitions, the agent loop, date-gated data sources |
| `models/` | Features, training, prediction, committed weights (`win_probability.json`) |
| `eval/` | Replay harness, three-arm experiment, odds cross-check, per-game results |
| `scripts/` | Snapshot gate, test-set construction, odds allowlist, game-log fetch |
| `tests/` | 64 tests — date gating, model contract, snapshot gate |
| `ui/` | Two Streamlit front ends |
| `data/` | Collection scripts, feature engineering, raw and sample datasets |
| `docs/` | This report, the session handoff, tool contracts, design notes |
