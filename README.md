# NBA Game Intelligence Agent

**CECS 499 capstone · Summer 2026 · University of Tennessee, Knoxville**

A system that predicts NBA games without ever being allowed to see the future — and an
experiment testing whether an LLM agent that *explains* the prediction makes it better.

It does not. That is the interesting part.

**Team:** Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

📄 **[Full report → `docs/REPORT.md`](docs/REPORT.md)** · 🔧 **[Session handoff → `docs/HANDOFF.md`](docs/HANDOFF.md)** · 🤖 **[Model handoff → `models/README.md`](models/README.md)**

---

## Table of contents

- [The 60-second version](#the-60-second-version)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Why we designed it this way](#why-we-designed-it-this-way)
- [The tools](#the-tools)
- [Results](#results)
- [Anticipated questions](#anticipated-questions) ← *predicted advisor Q&A*
- [What is not done](#what-is-not-done)
- [Repository map](#repository-map)

---

## The 60-second version

Pick a matchup, pick a date you are asking **from**, and the system produces a pregame
report — win probability, what drove it, who is hurt, and what it could not find out.
Everything it says is built only from what was knowable on the date you chose.

That constraint is the whole project. The 2025-26 season has already happened, so any
online LLM might simply *remember* who won. An accuracy number produced that way measures
memory, not prediction. So we gate the data twice, pin the language model's knowledge
cutoff to before the test window, and score everything with plain Python against a file no
agent tool can reach.

**Headline numbers** — all 1,322 games of 2025-26, a season the model never trained on:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **our model** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

**Headline finding** — we predicted that an agent handed the model's number would beat the
model alone. It lost. When the agent overruled the number, it was wrong **15 times out of
19** (pooled over two paired 40-game samples; two-sided sign test p ≈ 0.019).

---

## Quick start

Python 3.11+. Nothing here needs an API key.

```bash
git clone https://github.com/joshcannonai/nba-game-intelligence-agent.git
cd nba-game-intelligence-agent

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m models.train        # fit the model            (~7s)
pytest                        # 73 tests                 (~80s)
streamlit run ui/app.py       # the UI                   → localhost:8501
```

That is the whole setup. Verified from a clean clone and a fresh virtualenv on 2026-08-03.

<details>
<summary><b>Everything else you can run</b></summary>

```bash
# The report, deterministic, no LLM, instant
python -m agent.run --dry-run --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30

# Which tools have data, and where the missing inputs come from
python -m agent.run --status --source real

# The experiment: arm A + both baselines over all 1,322 games (~3s)
python -m eval.three_arms

# All three arms on a paired sample (needs `ollama serve`, ~50 min)
python -m eval.three_arms --arms abc --sample 40 --model ollama

# Materialise the pre-gated data directory the advisor asked for
python -m scripts.gate_snapshot --as-of 2026-01-14

# Confirm the odds file really holds CLOSING lines, not opening lines
python eval/crosscheck_odds.py

# The conversational UI over the same tools and the same gate
streamlit run ui/chat.py

# The real agent loop, local model, no API key (~40s/game)
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```

**Sharing it with someone else.** `pip install -r requirements.txt` then
`streamlit run ui/app.py` is the whole story — no API key, no database, no services. The
model weights (`models/win_probability.json`) and the sample data are committed, so a
fresh clone is immediately runnable. For a teammate on another machine,
`streamlit run ui/app.py --server.address 0.0.0.0` serves it on the local network.

The LLM paths are the only exception: `--model ollama` needs `ollama serve` running and
`ollama pull gemma4` once; `--model anthropic` needs `ANTHROPIC_API_KEY` in a `.env`.
Everything shown in the UI runs without either.
</details>

---

## How it works

```
   you pick a game        ┌────────────────────────────────────────┐
   and a date D    ─────▶ │  GATE 1 — scripts/gate_snapshot.py     │  plain Python.
                          │  copies data → data/snapshots/D/       │  no AI involved.
                          │  future results physically absent      │
                          └───────────────────┬────────────────────┘
                                              │
                          ┌───────────────────▼────────────────────┐
                          │  GATE 2 — agent/sources.py             │  per-tool precision
                          │  every read filtered to ≤ D            │
                          └────────┬──────────────────┬────────────┘
                                   │                  │
                    ┌──────────────▼─────┐   ┌────────▼────────────────┐
                    │  7 agent tools     │   │  8 features, as of D    │
                    │  agent/tools.py    │   │  models/features.py     │
                    └──────────┬─────────┘   └────────┬────────────────┘
                               │                      │
                    ┌──────────▼─────────┐   ┌────────▼────────────────┐
                    │  LLM agent loop    │   │  logistic regression    │
                    │  arms B and C      │   │  arm A                  │
                    └──────────┬─────────┘   └────────┬────────────────┘
                               └──────────┬───────────┘
                                          │
                          ┌───────────────▼────────────────────────┐
                          │  eval/three_arms.py                    │  reads results ONLY
                          │  accuracy · log loss · Brier           │  to score, after the
                          │  vs always-home · vs Vegas             │  prediction is made
                          └────────────────────────────────────────┘
```

Two halves answer the same question by different means.

**The model** (`models/`) is a logistic regression over eight features — rolling form,
win percentage, rest, back-to-backs, injury load — all computed strictly from games
*before* the one being predicted. It returns a win probability and nothing else.

**The agent** (`agent/`) is an LLM with seven tools. It decides what to retrieve for a
specific matchup, calls the tools, and writes a report in plain language. It has no
database, no web access, and no way to invent a number.

Both read through the same gated accessors, which is what makes comparing them fair.

### The three gates

| | What it stops | Where |
|---|---|---|
| **1. On-disk snapshot** | Future data reaching *anything* | `scripts/gate_snapshot.py` |
| **2. Query-time filter** | Future data reaching a *specific tool* | `agent/sources.py` |
| **3. Model knowledge cutoff** | The LLM *remembering* the result | Gemma 4, cutoff ~Jan 2025 |

Gate 1 is the one the advisor asked for by name on 2026-07-28: *"you're not relying on the
LLM to gate its own data. You pre-gate it."* Running it prints exactly what it removed:

```
$ python -m scripts.gate_snapshot --as-of 2026-01-14
Snapshot as of 2026-01-14 -> data/snapshots/2026-01-14

  samples/game_logs_2026.csv                       1,322 kept      719 outcomes cleared
  samples/odds_only.csv                           23,714 kept      726 dropped
  raw/injury_pst_2025_2026/injury_data.csv         2,272 kept    1,309 dropped
  raw/nba_stats_1947_present/Player Per Game.csv  32,606 kept      733 dropped
  ...
```

---

## Why we designed it this way

Six decisions that were not obvious, and the reasoning behind each.

### 1. Two gates, not one

They look redundant. They are not. **A snapshot can only be as strict as its loosest
legitimate reader** — `team_form_as_of` needs games strictly *before* the as-of date, while
`schedule_context` needs games *through* it. One on-disk cut cannot serve both without
starving one. So the snapshot removes what *nobody* may see, and the query-time filter
draws the finer per-tool line.

### 2. Future games keep their row; only the result is erased

The obvious implementation — drop future rows — is wrong. The agent is being asked to
preview a game that has not been played, so it must be able to see the game *exists*. What
it must not see is how it *ended*. So `gate_snapshot.py` keeps the row and blanks
`home_pts`, `away_pts`, `winner`.

### 3. The betting line was taken away from the agent

The agent had a `retrieve_betting_line` tool whose docstring said, in capitals, that the
line was context only. Running live on 2026-01-14, it wrote this into its own key factors:

> *"The closing betting line favors the home team, ORL (-5.5 spread)"*

The closing line is the benchmark we grade ourselves **against**. An agent that reads the
market and repeats it scores beautifully and has predicted nothing.

Nothing was violated in a dramatic way — the agent was asked what drove its reasoning and
truthfully reported a factor. The problem is that the tool made the shortcut *available*.
**Telling a model not to peek is a request. Removing the tool is a guarantee.** So the tool
is gone, `agent.sources.closing_line` remains for scoring only, and a test asserts the tool
cannot come back.

### 4. Placeholders report their gaps instead of guessing

A tool whose input does not exist returns
`{"status": "awaiting_input", "needs_from": ..., "needs": ...}`, and the agent is told to
report the gap rather than fill it. Never zero — an unknown injury list is not "nobody is
hurt."

The side effect is that the project's own blocking list is generated from the code:
`python -m agent.run --status --source real` prints what is built, what is stubbed, and
whose input each gap is waiting on.

### 5. Logistic regression, not a tree ensemble

Chosen for three properties XGBoost would not have given us: the weights are readable in a
pull request, it serialises to a few hundred bytes of named numbers instead of a pickle,
and it loads without `sklearn` — so the agent, the harness and the UI all run without the
training dependency installed.

It also keeps us honest. A fitted model with visible coefficients can be argued with. On
raw accuracy it barely beat the hand-tuned heuristic it replaced (66.5% vs 66.3%); what it
bought was calibration and *checkability*.

### 6. Split by season, never by random shuffle

A random shuffle lets a model learn from March in order to predict January, which inflates
accuracy by a few points that vanish the moment anyone checks. `TRAIN_SEASONS = (2024, 2025)`,
`TEST_SEASON = 2026`, and three tests fail if the test season is added to training.

The same trap sits inside feature construction: the natural way to compute a season win
percentage — group the season, take the mean — silently includes the game being predicted.
`models/features.py` advances every accumulator *after* emitting the row.

**Both rules are mutation-tested.** We broke each on purpose and confirmed tests caught it:

| Mutation | Tests that failed |
|---|---|
| Accumulators advanced before the row was emitted | 3 |
| Form window drifted out of sync with the agent's accessor | 1 |
| Test season added to the training seasons | 3 |

---

## The tools

Seven functions are the agent's entire world. Every retrieval tool takes an `as_of_date`.

| # | Tool | Status | Returns |
|---|---|---|---|
| 1 | `retrieve_matchup_context(matchup_id, as_of_date)` | ✅ | Team ratings, rest, injuries, head-to-head as of a date |
| 2 | `retrieve_player_splits(player_name, back_to_back)` | ✅ | Season averages, optional fatigue split |
| 3 | `retrieve_team_form(team_abbr, as_of_date, last_n)` | ✅ | Rolling 10-game record and point differential |
| 4 | `retrieve_injuries(team_abbr, as_of_date)` | ✅ | Who was known to be out that morning |
| 5 | `predict_win_probability(home, away, as_of_date)` | ✅ | Logistic regression output — **withheld in arm B** |
| 6 | `retrieve_schedule(as_of_date, days_ahead)` | ⏳ | Blocked on a committed forward-looking schedule table |
| 7 | `predict_stat_line(player, matchup_id, as_of_date)` | ⏳ | Points / rebounds / assists — not started |

Live status: `python -m agent.run --status --source real`

### The rules each tool follows

Every tool has a **skill** — a Markdown file in [`skills/`](skills/) saying when to call
it and what to do with the answer. The agent loads them into its system prompt at
startup, so editing one changes behaviour with no code change. That is deliberate: the
domain rules are the part teammates need to edit, and they should not have to open
`agent/run.py` to do it.

The block is built from the tools the agent was actually given, so arm B never receives
rules for a tool it does not have — `tests/test_skills.py` asserts the two arms' blocks
differ by exactly one section.

The most important rule is the one we could not write. We tried to encode "a >20 ppg
player is out, so drop the odds by N%" and could not find an N the data supports:
comparing each team against **itself**, with its top scorer and without, the difference
is **+0.0% (se 3.3%, n = 21 teams)**. A pooled comparison looks significant (+5.6%) and
points the wrong way, because having a star is a property of good teams. So the injury
skill tells the agent to report the list and let the fitted model price it. Reproduce
with `python -m eval.injury_impact`.

A review copy for the team is generated by `python scripts/skills_doc.py` →
[`docs/SKILLS.md`](docs/SKILLS.md).

<details>
<summary><b>The three tools that were cut, and why</b></summary>

The original design had ten. A scope cut nobody can explain later just looks like abandoned
work, so:

| Cut | Reason |
|---|---|
| `retrieve_news` | No source with reliable publication timestamps was ever found. Highest effort of the ten, least measurable contribution. Cut on merit. |
| `predict_best_player` | Depended entirely on `predict_stat_line`, which never started. A placeholder behind a placeholder. |
| `retrieve_betting_line` | **Not a scope cut — a leak.** See design decision 3 above. |
</details>

---

## Results

### The experiment

Three arms differing by **exactly one tool**, with a test enforcing that the two agent tool
lists differ by that single entry and nothing else. The difference *is* the measurement.

| Arm | What it is | Has `predict_win_probability`? | LLM? |
|---|---|---|---|
| **A** | Model only | — (*is* the model) | No |
| **B** | Agent only | No | Yes |
| **C** | Agent + model | Yes | Yes |

**Hypothesis, stated before the run: C beats both A and B.**

### Arm A, full season (n = 1,322)

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **arm A — logistic regression** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

Train accuracy 66.8% vs test 66.5% — a **generalisation gap of +0.3%**. Not overfit.

### All three arms, paired samples (n = 40 each)

| arm | seed 0 | seed 1 |
|---|---|---|
| **A — model only** | **75.0%** | **70.0%** |
| B — agent only | 57.5% | 62.5% |
| C — agent + model | 55.0% | 62.5% |
| Vegas | 57.5% | 77.5% |
| always-home | 50.0% | 62.5% |

Arms B and C call a language model once per game (~38s), so the full season across all
three arms is roughly 30 hours. They run on a fixed random sample instead, and **every arm
is scored on the same games**. The harness refuses a full-season LLM run rather than
quietly starting a 30-hour job.

**Do not quote arm A's 75% as our accuracy.** The season-long figure is 66.5%; these 40
games happened to suit it. Vegas scoring 57.5% on one sample and 77.5% on the other, in the
same season, shows how noisy n=40 is — which is exactly why the headline finding below is a
*paired* one.

### The finding: our hypothesis was wrong

Comparing headline accuracies asks whether a particular 40 games flattered one arm. The
paired question is immune to that: **when the agent overruled the model it was handed, how
often did overruling help?**

| sample | agreed | overrides | model right | agent right |
|---|---|---|---|---|
| seed 0 | 28 | 12 | 10 | 2 |
| seed 1 | 33 | 7 | 5 | 2 |
| **pooled** | **61** | **19** | **15** | **4** |

Two-sided sign test on the pooled overrides: **p ≈ 0.019**. (The harness prints a one-sided
value, 0.0096; since our pre-stated hypothesis pointed the *other* way, the two-sided
figure is the one to quote. The conclusion survives either.)

It was not overruling at the margins. The largest reversals took a confident correct call
and inverted it:

| game | model | agent | actual |
|---|---|---|---|
| CHI-ORL-2025-12-01 | 0.815 | 0.249 | home won |
| IND-PHI-2026-01-19 | 0.741 | 0.242 | home won |

**Why this matters.** The agent is not adding noise around a good estimate — it is
systematically talking itself out of the model's confident correct calls. Our leading
hypothesis is that it over-weights the injury list from `retrieve_injuries`: four names on
a list reads as decisive to a language model, whereas the fitted model has learned roughly
what a given injury load is actually worth. The model has a calibrated prior; the agent has
a narrative one.

That points at a concrete redesign rather than a vague conclusion — the agent should
probably *annotate* the model's number rather than *replace* it.

We are careful about scope: what we have shown is that **this** agent, with **these** seven
tools, on **these** 79 paired games, degrades a good estimate. Not that LLM agents cannot
help in general.

---

## Anticipated questions

These are the questions Prof. Sadovnik is most likely to ask, predicted from his actual
advising sessions (he asks students to walk him through one model end to end, he probes
leakage relentlessly, and he warned this team directly: *"since you're using a lot of AI to
generate this stuff, there's a little bit of work required to make sure that you understand
how all the stuff that you put together works"*).

### Leakage and gating

**Q: Walk me through what happens, step by step, when I press the predict button. Where exactly does the gating happen?**

Two places. First `scripts/gate_snapshot.py` runs — plain Python, no model involved — and writes a filtered copy of the data to `data/snapshots/<date>/`. Then the agent runs with `NBA_SNAPSHOT_DIR` pointed at that directory, so the future is not on disk at all. Second, `agent/sources.py` filters every individual read against the as-of date. In the UI that first step is the "Pre-gate data on disk" checkbox, and when it is on the page reports exactly how many rows were dropped and how many results cleared.

**Q: Why do you need both? Isn't the second one redundant if the first one works?**

They do different jobs. A snapshot can only be as strict as its loosest legitimate reader — `team_form_as_of` needs games strictly *before* the as-of date, but `schedule_context` needs games *through* it. One on-disk cut cannot serve both without starving one of them. So the snapshot removes what *nobody* may see, and the query-time filter draws the finer per-tool line.

**Q: If a game hasn't been played yet, why is it still in your snapshot at all?**

Because the agent is being asked to preview a game that has not happened, so it has to be able to see that the game *exists*. Dropping the row would hide the question along with the answer. `gate_snapshot.py` keeps the row and erases three columns — `home_pts`, `away_pts`, `winner`. On a 2026-01-14 snapshot that clears 719 results while keeping all 1,322 scheduled games.

**Q: How do you know your tests actually catch leakage? A test that always passes proves nothing.**

We broke each rule on purpose and confirmed tests failed. Advancing the feature accumulators *before* emitting the row instead of after — 3 tests caught it. Drifting the form window out of sync with the agent's accessor — 1 test. Adding the test season to the training seasons — 3 tests. That table is in the report, and it is the reason I trust the suite rather than just the count.

**Q: The LLM has read the internet. How do you know it isn't just remembering who won?**

That is the one gating the data cannot fix, so the scored replays run on Gemma 4 locally, whose cutoff is around January 2025 — verified behaviourally, in that it knows the 2024 Finals and does not know 2025 or 2026. Every game in the 2025-26 test window postdates that. The consequence I should state up front: the 2023-24 and 2024-25 seasons are demonstrations of the mechanism, **not** valid evaluation games for the LLM arms. They are still fine for training the logistic regression, which has no world knowledge to leak.

> **Weak spot:** "Verified behaviourally" means we asked it questions, not that we have a published cutoff date we can cite. It is evidence, not proof. If he pushes, say so plainly.

**Q: Your injury data — when exactly did you know a player was out?**

This is our weakest gate and I would rather name it than have it found. The log records **transaction dates** — when a player was placed on or activated from the injured list — not the moment the news broke. An as-of query on the morning of a game can therefore see a same-day placement that a real user might not have had. The size of that effect is unquantified.

> **Weak spot:** Genuinely unresolved. The honest answer is "it is a residual leak, probably small, and we did not measure it."

**Q: Your tests run in the same process as your code. How do you know a real run actually reads the snapshot and not the repo?**

That was the concern, so one test spawns an actual subprocess with `NBA_SNAPSHOT_DIR` set and asserts the child process resolves its data directory to the snapshot — `test_subprocess_actually_reads_the_snapshot`. Its docstring says what it is for: it guards the test below it from passing for the wrong reason. The test below then runs the same matchup twice, once against the snapshot and once against the full data, and asserts the reports are identical. The snapshot physically cannot leak, so if the query-time filter were leaking, those two would disagree.

**Q: What happens if I ask about a date past the end of your injury data? Does it just say nobody is hurt?**

No, and that case has its own test — `test_injuries_past_the_end_of_the_log_warn_rather_than_report_nobody_hurt`. Past the end of the log the payload carries a warning containing "UNKNOWN". An empty injury list and an unknown injury list are different facts, and reporting the second as the first is the exact failure the whole "null is never zero" rule exists to prevent.

**Q: Your injury data is two files stitched together. What happens at the join?**

They are kept as separate files so provenance survives, but they have to behave as one continuous log. A test asserts both sides of the seam are present — 2025-01-12 from the first file and 2025-01-13 from the second — and that the two files share no dates at all, since an overlap would double-count a transaction. Worth knowing: the older file contains one genuine duplicate of its own, so the test checks the files are disjoint *in time* rather than that every row is unique.

**Q: You gate everything else. Why isn't rest gated?**

Because rest is not knowledge about the future, it is knowledge about the calendar. The NBA publishes its full schedule in August, so "BOS plays the 23rd and the 25th" is knowable on any as-of date in that season — gating it would cripple a legitimate feature to defend against nothing. The *outcome* of the game on the 23rd is a different matter and is gated. `test_rest_is_schedule_based_not_as_of_gated` and `test_h2h_results_are_gated_even_though_rest_is_not` pin both halves of that distinction.

### The model

**Q: Walk me through one model. What's the input and what's the output?**

Input is three things: home team abbreviation, away team abbreviation, and an as-of date. From those, `models/features.py` builds an 8-element vector — the difference in win percentage, the difference in rolling 10-game point margin, rest difference, two back-to-back flags, injury-load difference, and each team's games played. Output is a single float, `home_win_prob`, between 0 and 1. It is a logistic regression, so the prediction is a weighted sum of those eight numbers pushed through a sigmoid. The weights are committed as readable JSON in `models/win_probability.json`.

**Q: Why logistic regression and not XGBoost?**

Three reasons, and none of them is that it was easier. The weights are readable in a pull request, so anyone can argue with the model. It serialises to a few hundred bytes of named numbers instead of a pickle. And it loads without `sklearn`, so the agent, the eval harness and the UI all run without the training dependency installed. XGBoost was the plan and the interface is built so it drops straight in — `models/README.md` documents that swap.

**Q: How did you split train and test, and why that way?**

Trained on 2023-24 and 2024-25, tested on 2025-26 — split by **season**, not by random shuffle. A random shuffle lets the model learn from March in order to predict January, which inflates accuracy by a few points that vanish the moment anyone checks. Two constants in `models/train.py` encode it and three tests fail if the test season enters the training set.

**Q: 66.5% — is that good? Compared to what?**

Against 55.5% for simply always picking the home team, and 69.0% for the Vegas closing line. Accuracy alone is close to meaningless on NBA games, so we also report log loss (0.6118 vs 0.5782 for Vegas) and Brier (0.2116 vs 0.1977). We are 2.5 points behind the market, which is where I would expect to be — the market has injury beat reporters and real money.

**Q: How do you know it isn't overfit?**

Train accuracy is 66.8%, test accuracy 66.5% — a generalisation gap of +0.3%. If it were memorising the training seasons that gap would be much larger.

**Q: If I swap home and away, what should your model do?**

It should *not* return mirrored probabilities, and there is a test for that. Home advantage is real, so p(BOS home vs ORL) + p(ORL home vs BOS) has to come out above 1.0. If those two summed to exactly 1.0 the model would have lost its home term entirely — and home court is worth roughly the whole always-pick-home baseline, about 55%. It is a cheap sanity check that catches a whole class of feature-construction bug.

**Q: Two of your features are `home_games_played` and `away_games_played`, and they carry real weight. What are they actually measuring?**

Honestly, I am not sure they are measuring basketball. For any given game those two are nearly identical, so the model may be fitting a schedule artifact rather than a real signal. It is flagged in `models/README.md` as worth an ablation, and we did not run it.

> **Weak spot:** He is likely to spot this on the weights table alone. Concede it immediately rather than inventing a rationalisation — "we should drop them and re-score" is the right answer.

### The agent and the negative result

**Q: You had a betting-line tool and now you don't. What happened?**

It caught the agent cheating, in the sense you have been describing all semester. Running live on 2026-01-14 the agent wrote *"The closing betting line favors the home team, ORL (-5.5 spread)"* into its own key factors. The closing line is the thing we grade ourselves against, so an agent that reads the market and repeats it scores beautifully and has predicted nothing. Nothing was violated dramatically — it was asked what drove its reasoning and truthfully reported a factor. The tool just made the shortcut available. Telling a model not to peek is a request; removing the tool is a guarantee. So the tool is gone, `closing_line` remains for scoring only, and a test asserts it cannot come back.

**Q: What was your hypothesis, and what did you actually find?**

We predicted arm C — the agent given the model's number — would beat both the model alone and the agent alone. It lost. It scored *worse* than the number it was handed. The mechanism is the part I would point at: on the games where the agent overruled the model, it was wrong 15 times out of 19.

**Q: Forty games is not many. How do you know that isn't noise?**

Two ways. First, the comparison is paired — every arm is scored on the same games, so a sample that happens to be easy or hard cancels out. Comparing headline accuracies would not survive n=40; asking "when the two disagree, who was right" does. Second, we ran a second independent sample. Seed 1 on its own is *not* significant — seven overrides and a 5–2 split is ordinary luck — but the direction replicates, and pooled across 19 overrides it holds.

**Q: What p-value, and was it one-sided or two-sided?**

The harness prints a one-sided sign test, p ≈ 0.0096. The two-sided value is p ≈ 0.019. Since our pre-stated hypothesis pointed the *opposite* way to the effect we observed, a one-sided test in the observed direction would be post-hoc, so the two-sided figure is the one to quote. It is still significant at 0.05, so the conclusion survives either test.

> **Weak spot:** Get in front of this. If he asks "one-sided?" and the answer sounds improvised, it looks like p-hacking. The recovery is that both tests clear 0.05.

**Q: Why do you think the agent overrules the model?**

Our leading hypothesis is that it over-weights the injury list it can see through `retrieve_injuries`. Four names on a list reads as decisive to a language model, whereas the fitted model has learned from two seasons roughly what a given injury load is actually worth. The model has a calibrated prior about injuries; the agent has a narrative one. The two biggest reversals fit that shape — it took 0.815 and 0.741 predictions and inverted them to 0.249 and 0.242, and was wrong both times.

> **Weak spot:** This is a hypothesis, not a finding. We have not read the per-game `key_factors` and categorised them. It is the top item on the open list, and if he asks "did you check?" the answer is no, not yet.

**Q: So should we conclude LLM agents don't help with prediction?**

No, and I would not let the report say that. What we have shown is that **this** agent, with **these** seven tools, on **these** 79 paired games, degrades a good estimate. The useful reading is narrower and more actionable: the agent should probably *annotate* the model's number rather than be free to *replace* it. That is a concrete redesign the result points at.

### Evaluation methodology

**Q: Who is doing the scoring? Not the LLM, I hope.**

No — `eval/three_arms.py` is plain Python. Each arm produces a probability, and the script compares it against ground truth read from a file no agent tool can reach. An unparseable LLM answer is recorded as a skip, never coerced to 0.5, because coercing would drag an arm toward the baseline and flatter it.

**Q: Your Vegas baseline — is that actually the closing line, or the opening line?**

We checked rather than assumed, because the whole baseline rests on it. `eval/crosscheck_odds.py` compares our file against an independent source that labels opening and per-book lines explicitly: 9 of 10 sampled games are closer to closing. There is a second detail worth stating — the spread-to-probability conversion uses a residual sigma fitted on the same 1,322 games, which makes the baseline slightly *stronger* than it deserves. That is a conservative bar, not a neutral one.

**Q: Are you and Sarvesh predicting on the same games?**

Not yet, and that was your ask on the 28th. Our arm A covers all 1,322 games of 2025-26. Reconciling to a shared game list — probably starting a few weeks into the season so the rolling features exist for everyone — is open, and it is the thing I would prioritise for the comparison section of the report.

> **Weak spot:** A direct, dated instruction that is still outstanding. Do not dress it up; say it is open and name who is doing it and by when.

### Do you understand your own code?

**Q: You used a lot of AI to build this. Explain the logistic regression to me without looking at it.**

It is a weighted sum plus an intercept, pushed through a sigmoid to squash it into 0–1. Fitting it means choosing the weights that minimise log loss over the training games. Because the features are standardised before fitting, the weights are directly comparable — which is how I can say win-percentage difference (+0.396) and rolling-margin difference (+0.378) are doing most of the work, and rest difference (+0.003) is doing essentially none. Every sign is physically sensible: better record helps, being on a back-to-back hurts, injuries hurt, an opponent on a back-to-back helps.

**Q: Show me the line of code that would leak if you got it wrong.**

The accumulator update in `models/features.py`. The obvious way to compute a team's season win percentage is to group the season and take the mean — and that silently includes the game you are predicting, which gives you a very accurate and completely worthless model. Every accumulator there is advanced *after* the row is emitted. That is the mutation we broke first, and three tests caught it.

**Q: You have two implementations of rolling form — one in the agent, one in the model. Why isn't that a bug waiting to happen?**

It is, which is why there is a test for it. The agent's `team_form_as_of` re-scans the log per query; the model's version walks each season once carrying accumulators forward, because it needs to do 3,777 games times two teams. Same idea, different code, so `tests/test_model_contract.py` asserts they agree game for game. If they ever drift, that test fails rather than the model quietly training on a different world than the agent sees.

### Scope, teamwork, and what's missing

**Q: Your proposal promised projected stat lines and a best-player pick. Where are they?**

Not built. `predict_stat_line` exists as a tool signature and returns `awaiting_input`; the regression behind it never started. `predict_best_player` depended on it, so it was cut — a placeholder behind a placeholder. `retrieve_news` was cut too, because no source with reliable publication timestamps was ever found. The system reports these gaps rather than hiding them: `python -m agent.run --status` prints what is built and what is blocked, generated from the code.

> **Weak spot:** This is a stated deliverable that does not exist, and "the status board reports it" is a good process answer to a scope question, not a substitute for the feature.

**Q: You built the win model. Wasn't that Sarvesh's piece?**

Yes, and I want to be straight about it. The model interface was mine to build either way, and I filled in a working baseline behind it so nothing downstream was blocked — the eval harness had nothing to score without one. `models/README.md` is written as a handoff to him, the swap is one file, and beating 66.5% is the open task. It was not meant to take his work over.

> **Weak spot:** The most awkward question on this list. Answer it in one honest breath and move to what Sarvesh does next; do not over-explain, and do not disparage his progress.

**Q: If I clone the repository right now, do I get what you just showed me?**

Not from `main` — that is the one thing I need to fix before the final submission. The current work sits on a branch that is 21 commits ahead, and `main` has no `models/`, no eval harness, and no snapshot gate. Landing that PR is the top item on my list.

> **Weak spot:** If he clones `main` before this is merged, he sees roughly half a project. Land it first.

### Ethics and AI use

**Q: How much of this did you write, and how much did the AI write?**

AI assistance was used throughout, and it is disclosed in §13 of the report rather than concealed. The constraint I held to is that every line merged is one I can explain and defend — which is what this conversation is testing. The architectural calls are mine and each has a reason behind it: two gates rather than one, keeping future rows while erasing outcomes, removing the betting-line tool instead of instructing around it, logistic regression over a tree ensemble, splitting by season.

**Q: Does the OpenAI sandbox-escape incident connect to anything you ran into?**

Directly, and it is the reason §5.3 is written the way it is. The agent had a tool it was told in capital letters not to lean on, and it leaned on it — not maliciously, just because the shortcut was available and it had no concept that consulting the answer key was off-limits. That is the same shape as a system deciding the fastest route to "write an exploit" is to go find the published solutions. The lesson we took is the one you described: you cannot fix it with an instruction, you fix it by making the shortcut structurally unavailable. That is why the gating is a separate Python step and not a prompt.

---

### The five questions most likely to land a punch

Ranked by exposure, with how to handle each.

1. **"Wasn't the model Sarvesh's job?"** — The most awkward, because the honest answer touches a teammate's delivery. One breath: the interface was mine, nothing downstream could be scored without a baseline behind it, `models/README.md` is the handoff, beating 66.5% is his task. Then move on. Do not editorialise about his progress.

2. **"Where are the projected stat lines?"** — A stated PDP deliverable that does not exist. Do not lead with the status board; lead with "not built," then explain that the tool signature is waiting and the dependency chain that cut `predict_best_player` with it.

3. **"Explain the logistic regression without looking."** — He does this to every student and he did it to Sarvesh on the 28th. Rehearse it out loud: weighted sum, sigmoid, fit by minimising log loss, standardised features so weights compare, and be ready to name which two features dominate and which is doing nothing.

4. **"One-sided or two-sided?"** — The headline finding rests on a sign test that the code prints one-sided while the pre-stated hypothesis pointed the other way. Volunteer the two-sided number (p ≈ 0.019) *before* he asks. Getting there second looks like p-hacking; getting there first looks rigorous.

5. **"Are you and Sarvesh on the same games yet?"** — A direct instruction from 2026-07-28 that is still open. Answer with a plan and a date, not an explanation.

**Two more worth rehearsing:** the `home_games_played` weight (concede it may be a schedule artifact), and whether `main` reflects what you demoed (land PR #17 first, or the answer is embarrassing).

---

## What is not done

Stated plainly, because the gaps we name are less dangerous than the ones we do not.

| | Item | Owner |
|---|---|---|
| 1 | **`predict_stat_line`** — "projected stat lines" was a stated PDP deliverable. The tool signature exists and returns `awaiting_input`; the regression behind it does not. | Sarvesh |
| 2 | **`retrieve_schedule`** — blocked on a committed forward-looking schedule table. `data/raw/` stopped being gitignored on 7/21, so nothing is in the way. | Patrick |
| 3 | **Why the agent overrules the model** — the most valuable open question. Needs error analysis of the 19 overrides' `key_factors`. | open |
| 4 | **Larger n for arms B and C** — 40 games is a ±8% band. | open |
| 5 | **The "let it cheat" ablation** — the advisor suggested deliberately un-gating the agent as a contrast condition. Never run. | open |
| 6 | **Apples-to-apples with Sarvesh's model** — the advisor asked that both predict on the same game set. | Sarvesh |

Known weaknesses in what *is* built are documented in [`docs/REPORT.md` §11](docs/REPORT.md)
and [`models/README.md`](models/README.md) — including that injury data are transaction
dates rather than news timestamps, that injury importance over-penalises, that
`home_games_played` may be fitting a schedule artifact, and that there is no
opponent-adjusted strength of schedule.

---

## Repository map

| Path | Contents |
|---|---|
| `agent/` | Tool definitions, the agent loop, date-gated data sources, skill loader |
| `skills/` | One Markdown file per tool — the rules the agent follows |
| `models/` | Features, training, prediction, committed weights — see [`models/README.md`](models/README.md) |
| `eval/` | Replay harness, three-arm experiment, odds cross-check, per-game results |
| `scripts/` | Snapshot gate, test-set construction, odds allowlist, game-log fetch |
| `tests/` | 73 tests — date gating, model contract, snapshot gate, skills |
| `ui/` | Two Streamlit front ends |
| `data/` | Collection scripts, feature engineering, raw and sample datasets |
| `docs/` | [Report](docs/REPORT.md) · [Handoff](docs/HANDOFF.md) · [Tool contracts](docs/tool-contracts.md) · [Design notes](docs/agent-design-notes.md) |

---

## Working agreements

- Python 3.11+. Dependencies in `requirements.txt`.
- Feature branches and pull requests. No direct pushes to `main`.
- Never commit API keys. Copy `.env.example` to `.env` locally (gitignored).
- `langchain` must be the **1.x** line — `agent/run.py` calls `create_agent`, which does not
  exist in 0.3, and a 0.3 install next to a 1.x `langchain-core` fails to import at all.
- AI-assisted code is fine per course policy, but you own and can explain every line you
  merge. Disclosure is in [`docs/REPORT.md` §13](docs/REPORT.md).
