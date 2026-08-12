# NBA Game Intelligence Agent

**CECS 499 capstone · Summer 2026 · University of Tennessee, Knoxville**

A system that predicts NBA games without being allowed to see the future, plus an
experiment that tests whether an LLM agent changes the predictor's result.

**Team:** Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

📄 **This README is the current submission contract** · 🗄️ **[Superseded report draft → `docs/REPORT.md`](docs/REPORT.md)** · 🤖 **[Model handoff → `models/README.md`](models/README.md)**

---

## Table of contents

- [The 60-second version](#the-60-second-version)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Why we designed it this way](#why-we-designed-it-this-way)
- [The tools](#the-tools)
- [Results](#results)
- [Submission checks](#submission-checks)
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

**Verified Model A numbers:** all 1,322 games of 2025-26, a season the model never
trained on:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **our model** | **65.9%** | **0.6150** | **0.2130** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

Results for Models B and C are not quoted here yet. Their current prompts must be run
through the actual UI endpoint over the same complete season before those numbers are
treated as evidence.

---

## Quick start

Python 3.11+. Nothing here needs an API key.

```bash
git clone https://github.com/joshcannonai/nba-game-intelligence-agent.git
cd nba-game-intelligence-agent

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m models.train        # fit the win model        (~7s)
pytest                        # full Python verification suite

# The site. The front end is built output, which is gitignored, so build it once.
cd ui/web && bun install && bun run build && cd ../..
python -m ui.serve            # → localhost:8000
```

Without the build step `python -m ui.serve` still runs, but it serves the API only and
tells you so. Everything else in this README works without touching the front end.

That is the whole setup. `python -m eval.three_arms` reproduces Model A's 65.89%
season accuracy without any further steps.

**Windows note.** Fixed on 2026-08-04. If you are on a checkout older than that and see
`UnicodeDecodeError: 'charmap' codec can't decode byte ...`, that is Python on Windows
defaulting to cp1252 while our data contains accented player names. Pull `main`.

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

# Submission evaluation: every one of the 1,322 games through the actual UI paths.
# A calls /api/predict. B and C consume /api/run's SSE stream with Gemma 4,
# the runtime system prompt, skill files, tool calls, and gate receipts.
# In another terminal first: ollama serve; ollama pull gemma4; python -m ui.serve
python -m eval.ui_agent_eval --full-season --arms ABC

# Materialise the pre-gated data directory the advisor asked for
python -m scripts.gate_snapshot --as-of 2026-01-14

# Confirm the odds file really holds CLOSING lines, not opening lines
python eval/crosscheck_odds.py

# The presentation site: agent, three-arm comparison, and system evidence
python -m ui.serve                       # → localhost:8000

# Remove one tool and re-run the same games -- what was that tool worth?
python -m eval.ablation --sample 50 --arm B --model ollama

# Regenerate the site's data after retraining a model or re-running an eval
python -m scripts.build_site_data
python -m scripts.build_site_data --check   # what the test suite runs

# Fit the stat-line model (needs data/raw/player_box_scores_prior/)
python -m models.train_stat_line

# The conversational UI over the same tools and the same gate
streamlit run ui/chat.py

# The real agent loop, local model, no API key (~40s/game)
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```

**Editing the site.** The front end is `ui/web/src/App.tsx` and `index.css`, one file
each. Everything it displays comes from `ui/web/src/data/*.json`, which is generated by
`python -m scripts.build_site_data` from the model files and the skill
files. **Do not edit those JSON files by hand**: they are committed only because a static
build cannot read the repo at run time, and `tests/test_site_data.py` fails when they stop
matching what the repo produces. Retrain a model or rerun an arm, then regenerate, then
`bun run build`.

**The site.** `python -m ui.serve` serves the built front end and the agent from one
process at `localhost:8000`. The presentation view has three tabs: run any of the three
arms on any of the 1,322 games, inspect the measured three-arm comparison, and review the
system details. Add `?details=1` to expose the Prompt, Tools, and Data reference tabs for
deeper review without cluttering the live presentation. A read-only copy is
hosted at https://nba-agent-cecs499.vercel.app -- everything works there except the Agent
tab, which needs a local model and says so, which means you do not need to build anything
to look at the results.

**Sharing it with someone else.** `pip install -r requirements.txt` then
`python -m ui.serve` is the whole story — no API key, no database, no services. The
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
legitimate reader** — `team_form_as_of` needs game outcomes *through* the as-of date, while
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
raw accuracy it barely trails the earlier hand-tuned heuristic (65.9% vs 66.3%); what it
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
| 2 | `retrieve_player_splits(player_name, as_of_date, back_to_back)` | ✅ | Prior-completed-season averages, optional fatigue split |
| 3 | `retrieve_team_form(team_abbr, as_of_date, last_n)` | ✅ | Rolling 10-game record and point differential |
| 4 | `retrieve_injuries(team_abbr, as_of_date)` | ✅ | Who was known to be out that morning |
| 5 | `predict_win_probability(matchup_id, as_of_date)` | ✅ | The same gated Model A output, available only to Model C |
| 6 | `retrieve_schedule(as_of_date, days_ahead)` | ✅ | Fixtures from the season game log. Teams and dates only, never a score |
| 7 | `predict_stat_line(player, matchup_id, as_of_date)` | ✅ | Points / rebounds / assists. Ridge on 2023-24, validated on 2024-25 |

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
| `predict_best_player` | Depended entirely on `predict_stat_line`, which was not built at the time. Cut then; `predict_stat_line` has since landed, so this could be revisited. |
| `retrieve_betting_line` | **Not a scope cut — a leak.** See design decision 3 above. |
</details>

---

## Results

### Current submission contract

The three arms are intentionally different in only one place:

| Arm | Runtime path | LLM | Predictor available to the agent? |
|---|---|---|---|
| **A** | `POST /api/predict` | No | A is the predictor |
| **B** | `POST /api/run` | Gemma 4 | No |
| **C** | `POST /api/run` | Gemma 4 | Yes, as one optional tool result |

Models B and C share the same reasoning core, retrieval skills, retrieval tools,
temperature, and local language model. Model C adds only `predict_win_probability`,
its tool schema, and its matching instructions. It may agree or
disagree with Model A, and the UI displays the actual system prompt, tool calls, tool
results, and gate receipts used in the run.

Model A has been re-run on every one of the 1,322 games through the same predictor path
used by the UI. It produced 871 correct predictions, or **65.89%** accuracy. Its test log
loss is 0.6150 and Brier score is 0.2130. These numbers differ from older reports because
the replay now stops date-only injury records at the previous calendar day, which removes
a same-day leakage risk.

The submission evaluator is `eval/ui_agent_eval.py`. In `--full-season` mode it runs the
same 1,322 games for A, B, and C, gives each the previous calendar day as its cutoff,
requires all five retrieval calls for B and those same calls plus the predictor for C,
requires passed gate receipts, and writes append-only checkpoints. The final outcome and
a reconstructed decimal price derived from the pre-tip closing spread are joined only
after the UI returns its prediction. The price is not a quoted moneyline.

Older sample evaluations are historical artifacts from earlier prompts. They are not
presented on the live site and should not be cited as current B or C performance.

---

## Submission checks

These are the questions Prof. Sadovnik is most likely to ask, predicted from his actual
advising sessions (he asks students to walk him through one model end to end, he probes
leakage relentlessly, and he warned this team directly: *"since you're using a lot of AI to
generate this stuff, there's a little bit of work required to make sure that you understand
how all the stuff that you put together works"*).

### Leakage and gating

**Q: Walk me through what happens, step by step, when I press the predict button. Where exactly does the gating happen?**

The actual UI evaluator enforces two runtime boundaries. The server first binds the run to one matchup and a strictly pregame cutoff, then every tool rejects any model-selected matchup, team, or cutoff that differs before touching its data source. `agent/sources.py` independently filters every historical read through that cutoff. `scripts/gate_snapshot.py` is an optional manual third layer that can materialize a filtered data directory for a fixed-date demo, but the full-season evaluator does not pretend to build 1,322 different snapshots.

**Q: Why do you need both? Isn't the second one redundant if the first one works?**

The server binding prevents the language model from changing the question. The source filter prevents an authorized query from returning later historical records. The optional snapshot is defense in depth for a fixed-date demonstration, not part of each full-season UI request.

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

**Q: 65.9% — is that good? Compared to what?**

Against 55.5% for simply always picking the home team and 69.0% for the Vegas closing line. Accuracy alone is incomplete, so we also report log loss (0.6150 vs 0.5782 for Vegas) and Brier (0.2130 vs 0.1977). We are 3.1 points behind the market, which is plausible because the market has better real-time injury information and real money.

**Q: How do you know it isn't overfit?**

Train accuracy is 66.6%, test accuracy 65.9%, a generalisation gap of 0.7 points. If it were memorising the training seasons that gap would be much larger.

**Q: If I swap home and away, what should your model do?**

It should *not* return mirrored probabilities, and there is a test for that. Home advantage is real, so p(BOS home vs ORL) + p(ORL home vs BOS) has to come out above 1.0. If those two summed to exactly 1.0 the model would have lost its home term entirely — and home court is worth roughly the whole always-pick-home baseline, about 55%. It is a cheap sanity check that catches a whole class of feature-construction bug.

**Q: Two of your features are `home_games_played` and `away_games_played`, and they carry real weight. What are they actually measuring?**

Honestly, I am not sure they are measuring basketball. For any given game those two are nearly identical, so the model may be fitting a schedule artifact rather than a real signal. It is flagged in `models/README.md` as worth an ablation, and we did not run it.

> **Weak spot:** He is likely to spot this on the weights table alone. Concede it immediately rather than inventing a rationalisation — "we should drop them and re-score" is the right answer.

### The agent experiment

**Q: You had a betting-line tool and now you don't. What happened?**

It caught the agent cheating, in the sense you have been describing all semester. Running live on 2026-01-14 the agent wrote *"The closing betting line favors the home team, ORL (-5.5 spread)"* into its own key factors. The closing line is the thing we grade ourselves against, so an agent that reads the market and repeats it scores beautifully and has predicted nothing. Nothing was violated dramatically — it was asked what drove its reasoning and truthfully reported a factor. The tool just made the shortcut available. Telling a model not to peek is a request; removing the tool is a guarantee. So the tool is gone, `closing_line` remains for scoring only, and a test asserts it cannot come back.

**Q: What was your hypothesis, and what did you actually find?**

Our hypothesis is that Model C, the agent given Model A as one extra data point, will beat both A and B. The current B and C prompts have not completed their full-season actual-UI run, so the submission does not claim that hypothesis succeeded or failed yet.

**Q: Forty games is not many. How do you know that isn't noise?**

It would be too small for a final claim. That is why the submission evaluator runs every arm on the same 1,322 games and checkpoints each actual UI result. Old 40-game samples remain historical artifacts, not current evidence.

**Q: What p-value, and was it one-sided or two-sided?**

We will calculate a paired two-sided test after the current full-season run finishes. Quoting the old prompt's sample p-value for the current agent would mix two different systems.

> **Weak spot:** No current p-value exists until the current B and C run is complete.

**Q: Why do you think the agent overrules the model?**

That is an analysis question for the completed per-game output. The evaluator keeps each final probability and `key_factors`, so disagreements can be categorized after the run without guessing from aggregate accuracy.

> **Weak spot:** This is a hypothesis, not a finding. We have not read the per-game `key_factors` and categorised them. It is the top item on the open list, and if he asks "did you check?" the answer is no, not yet.

**Q: So should we conclude LLM agents don't help with prediction?**

No. The experiment measures this prompt, this local model, these tools, and this gated season. It cannot support a general claim about LLM agents.

### Evaluation methodology

#### Verified actual-UI workbook

[`NBA-Actual-UI-Agent-Evaluation-Shared-10-Games.xlsx`](docs/evaluation/NBA-Actual-UI-Agent-Evaluation-Shared-10-Games.xlsx)
is the current manually assembled professor-review workbook. It records the
same 10 games for all three models using the fixed 2026-04-05 cutoff. Model A ran through
`POST /api/predict`; Models B and C ran through the website's `POST /api/run`
SSE path with `gemma4:latest` via Ollama. The workbook contains formula-linked
Summary, Model A, Model B, Model C, UI Trace, and Methodology sheets. Its 30
model rows are also committed as a [Git-diffable CSV](docs/evaluation/verified-actual-ui-results.csv),
with the artifact checksum and contract in the [manifest](docs/evaluation/manifest.json).

The verified sample result is A 5/10, B 4/10, and C 6/10. Every required B/C
gate receipt passed. This is a shared classroom comparison, not a claim of
season-level B/C accuracy. The older full-season workbook is not included
because its B/C rows predated the corrected actual-UI evaluation contract.

**Q: Who is doing the scoring? Not the LLM, I hope.**

No — `eval/three_arms.py` is plain Python. Each arm produces a probability, and the script compares it against ground truth read from a file no agent tool can reach. An unparseable LLM answer is recorded as a skip, never coerced to 0.5, because coercing would drag an arm toward the baseline and flatter it.

**Q: Your Vegas baseline — is that actually the closing line, or the opening line?**

We checked rather than assumed, because the whole baseline rests on it. `eval/crosscheck_odds.py` compares our file against an independent source that labels opening and per-book lines explicitly: 9 of 10 sampled games are closer to closing. There is a second detail worth stating — the spread-to-probability conversion uses a residual sigma fitted on the same 1,322 games, which makes the baseline slightly *stronger* than it deserves. That is a conservative bar, not a neutral one.

**Q: Are you and Sarvesh predicting on the same games?**

Yes for the verified shared comparison. All three models ran the same 10 games
provided by Sarvesh with the same fixed 2026-04-05 cutoff. The committed
workbook and CSV show that exact list. The separate full-season actual-UI run
is still incomplete, so we do not claim full-season Model B or C accuracy.

### Do you understand your own code?

**Q: You used a lot of AI to build this. Explain the logistic regression to me without looking at it.**

It is a weighted sum plus an intercept, pushed through a sigmoid to squash it into 0–1. Fitting it means choosing the weights that minimise log loss over the training games. Because the features are standardised before fitting, the weights are directly comparable — which is how I can say win-percentage difference (+0.396) and rolling-margin difference (+0.378) are doing most of the work, and rest difference (+0.003) is doing essentially none. Every sign is physically sensible: better record helps, being on a back-to-back hurts, injuries hurt, an opponent on a back-to-back helps.

**Q: Show me the line of code that would leak if you got it wrong.**

The accumulator update in `models/features.py`. The obvious way to compute a team's season win percentage is to group the season and take the mean — and that silently includes the game you are predicting, which gives you a very accurate and completely worthless model. Every accumulator there is advanced *after* the row is emitted. That is the mutation we broke first, and three tests caught it.

**Q: You have two implementations of rolling form — one in the agent, one in the model. Why isn't that a bug waiting to happen?**

It is, which is why there is a test for it. The agent's `team_form_as_of` re-scans the log per query; the model's version walks each season once carrying accumulators forward, because it needs to do 3,777 games times two teams. Same idea, different code, so `tests/test_model_contract.py` asserts they agree game for game. If they ever drift, that test fails rather than the model quietly training on a different world than the agent sees.

### Scope, teamwork, and what's missing

**Q: Your proposal promised projected stat lines and a best-player pick. Where are they?**

Built as of 2026-08-04. `predict_stat_line` is backed by ridge regressions on a player's trailing 5- and 10-game form, fitted on 2023-24 and validated on 2024-25, and never on the season being replayed. It beats a trailing 5-game average by 0.061 points of MAE, which is real and small, and the skill tells the agent to say so rather than imply otherwise. `predict_best_player` was cut when it was a placeholder behind a placeholder, and `retrieve_news` because no source with reliable publication timestamps was ever found. `python -m agent.run --status` prints what is built and what is blocked, generated from the code.

> **Weak spot:** This is a stated deliverable that does not exist, and "the status board reports it" is a good process answer to a scope question, not a substitute for the feature.

**Q: You built the win model. Wasn't that Sarvesh's piece?**

Yes, and I want to be straight about it. The model interface was mine to build either way, and I filled in a working baseline behind it so nothing downstream was blocked. `models/README.md` documents the handoff and the fitted weights remain replaceable through the same interface.

> **Weak spot:** The most awkward question on this list. Answer it in one honest breath and move to what Sarvesh does next; do not over-explain, and do not disparage his progress.

**Q: If I clone the repository right now, do I get what you just showed me?**

Yes after this submission PR is merged. The PR includes the runnable UI paths, evaluator, skill prompt assembly, date gates, model weights, and tests.

> **Weak spot:** If he clones `main` before this is merged, he sees roughly half a project. Land it first.

### Ethics and AI use

**Q: How much of this did you write, and how much did the AI write?**

AI assistance was used throughout, and it is disclosed in §13 of the report rather than concealed. The constraint I held to is that every line merged is one I can explain and defend — which is what this conversation is testing. The architectural calls are mine and each has a reason behind it: two gates rather than one, keeping future rows while erasing outcomes, removing the betting-line tool instead of instructing around it, logistic regression over a tree ensemble, splitting by season.

**Q: Does the OpenAI sandbox-escape incident connect to anything you ran into?**

Directly, and it is the reason §5.3 is written the way it is. The agent had a tool it was told in capital letters not to lean on, and it leaned on it — not maliciously, just because the shortcut was available and it had no concept that consulting the answer key was off-limits. That is the same shape as a system deciding the fastest route to "write an exploit" is to go find the published solutions. The lesson we took is the one you described: you cannot fix it with an instruction, you fix it by making the shortcut structurally unavailable. That is why the gating is a separate Python step and not a prompt.

---

### The five questions most likely to land a punch

Ranked by exposure, with how to handle each.

1. **"Wasn't the model Sarvesh's job?"** — Explain the interface and handoff in one breath, then show that both Model A and Model C call the same committed predictor.

2. **"Where are the projected stat lines?"** — Built. The honest framing is what it cost: the only player-level data in the repo covers the season being replayed, so it had to be fitted on two prior seasons scraped for the purpose, and it beats a trailing 5-game average by 0.061 points of MAE. Lead with the baseline, not the model.

3. **"Explain the logistic regression without looking."** — He does this to every student and he did it to Sarvesh on the 28th. Rehearse it out loud: weighted sum, sigmoid, fit by minimising log loss, standardised features so weights compare, and be ready to name which two features dominate and which is doing nothing.

4. **"Why are the old sample results gone?"** — Because the prompt and evaluation contract changed. Mixing old outputs with the current agent would be inaccurate.

5. **"Are all models on the same games?"** — Yes. Full-season mode selects the same ordered 1,322-game list for A, B, and C.

**Two more worth rehearsing:** the `home_games_played` weight (concede it may be a schedule artifact), and the difference between a completed code path and a still-running evaluation.

---

## What is not done

Stated plainly, because the gaps we name are less dangerous than the ones we do not.

| | Item | Status |
|---|---|---|
| 1 | ~~**`predict_stat_line`**~~ | Built. Fitted on 2023-24, validated on 2024-25. |
| 2 | ~~**`retrieve_schedule`**~~ | Built from the season game log. |
| 3 | **Complete the actual-UI full-season runs for Models B and C** | Evaluator is built and resumable; results are intentionally not claimed until every row finishes. |
| 4 | **Build the full-season formula-linked workbook from the completed actual-UI CSV** | The verified shared 10-game workbook is included. The full-season workbook remains pending the B and C rows. |
| 5 | **The "let it cheat" ablation** | Suggested by the advisor, not run. |

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
| `tests/` | Date gating, model contract, snapshot gate, skills, stat line, UI evaluation contract |
| `ui/` | Two Streamlit front ends |
| `data/` | Collection scripts, feature engineering, raw and sample datasets |
| `docs/` | [Verified evaluation workbook](docs/evaluation/NBA-Actual-UI-Agent-Evaluation-Shared-10-Games.xlsx) · [Report](docs/REPORT.md) · [Handoff](docs/HANDOFF.md) · [Tool contracts](docs/tool-contracts.md) · [Design notes](docs/agent-design-notes.md) |

---

## Working agreements

- Python 3.11+. Dependencies in `requirements.txt`.
- Feature branches and pull requests. No direct pushes to `main`.
- Never commit API keys. Copy `.env.example` to `.env` locally (gitignored).
- `langchain` must be the **1.x** line — `agent/run.py` calls `create_agent`, which does not
  exist in 0.3, and a 0.3 install next to a 1.x `langchain-core` fails to import at all.
- AI-assisted code is fine per course policy, but you own and can explain every line you
  merge. Disclosure is in [`docs/REPORT.md` §13](docs/REPORT.md).
