# NBA Game Intelligence Agent — Final Report

**CECS 499 · Senior Transdisciplinary Capstone · Summer 2026 · University of Tennessee, Knoxville**

**Team:** Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel
**Advisor:** Prof. Amir Sadovnik

**Repository:** `github.com/joshcannonai/nba-game-intelligence-agent`

---

> **Draft v1, 2026-08-03.** Sections 1–7 and 9–11 are written and every number in
> them was regenerated from a clean checkout today. Sections 8 and 12 are shells for
> Sarvesh, Patrick and Kirtan to fill in. Appendix A lists the command that produces
> each number.

---

## Contents

1. [Abstract](#1-abstract)
2. [Problem](#2-problem)
3. [Architecture](#3-architecture)
4. [Data sources](#4-data-sources)
5. [Date gating](#5-date-gating)
6. [The agent](#6-the-agent)
7. [The model](#7-the-model)
8. [Stat-line and win models — *shell*](#8-stat-line-and-win-models--shell)
9. [Experiment and results](#9-experiment-and-results)
10. [Discussion](#10-discussion)
11. [Limitations](#11-limitations)
12. [Team contributions — *shell*](#12-team-contributions--shell)
13. [AI use disclosure](#13-ai-use-disclosure)
    - [Appendix A — Reproducing the numbers](#appendix-a--reproducing-the-numbers)
    - [Appendix B — References](#appendix-b--references)

---

## 1. Abstract

We built a system that predicts NBA game outcomes and explains its reasoning, then ran
an experiment to test whether the explaining part helps.

The system has two halves. A logistic regression reads eight engineered features and
returns a win probability. An LLM agent calls seven retrieval tools, reasons over what it
finds, and writes a pregame report. Both are held to the same rule: every query carries
an *as-of date*, and nothing published after that date reaches either one.

That rule is what lets us test on the 2025-26 season. The season has already been played,
so a language model may simply remember the results. Without the constraint, any accuracy
number would measure memory rather than prediction.

On all 1,322 games of 2025-26, held out from training, the model scores 66.5% accuracy.
Always picking the home team scores 55.5%. The Vegas closing line scores 69.0%.

We expected the agent-plus-model combination to beat the model alone. It did not. When the
agent overruled the number it was given, it was wrong 15 times out of 19 (two pooled
40-game samples, two-sided sign test p ≈ 0.019). The explanation layer costs accuracy
instead of adding it. That result is the main finding of the project.

---

## 2. Problem

Predicting NBA games has an obvious benchmark. The betting market prices every game, and
the closing line is hard to beat. What makes the problem worth a capstone is not the
prediction itself. It is the evaluation.

To know whether a system works, you test it on games it has not seen. For a classical
model that is routine: hold out a season. For a system built on a large language model it
is a real hazard, because the model has read the internet. Ask it who won a game in
December 2025 and it may know.

Our advisor framed this as an alignment problem, using the OpenAI `exploit-gym` incident
as the example: a system optimising for a goal takes whatever route reaches it, and the
fastest route is often the one a person would call cheating. The model has no concept of
cheating. Instructions in a prompt do not fix that. The shortcut has to be made
unavailable.

So the real subject of this project is leakage control. Prediction is the vehicle.

### 2.1 Three kinds of leak

These get collapsed into one word. They need different defences.

| Leak | Example | Defence |
|---|---|---|
| Data | A tool returns an injury report filed after the game | Date gating (§5) |
| Model knowledge | The LLM remembers the result from training | Cutoff-pinned local model (§5.4) |
| Benchmark | The system reads the answer off what we grade it against | Tool removal (§6.3) |

We anticipated the first two. We did not anticipate the third, caught it happening, and
had to fix it with a design change rather than a rule. That is §6.3.

---

## 3. Architecture

The pipeline, given a game and a date **D**:

1. **`gate_snapshot.py`** copies the data into `data/snapshots/D/`, with future results
   removed. Plain Python, no AI.
2. **`sources.py`** filters every read to on-or-before **D**.
3. Two consumers read through that same filter: **7 tools** feed the agent (arms B and C),
   and **8 features** feed the model (arm A).
4. **`three_arms.py`** scores all three arms against the results, after the fact.

Two points about that pipeline matter more than the steps themselves.

The gate runs twice on purpose. The snapshot removes what nobody may see. The query-time
filter decides what each tool may see. §5.3 explains why one layer is not enough.

The model and the agent share their data access. `models/features.py` imports the same
gated accessors that the agent's tools use. If the model could see anything the agent
could not, the comparison in §9 would be measuring our plumbing rather than our ideas. A
test asserts the two implementations of rolling form agree game for game.

---

## 4. Data sources

Everything the running system reads, with row counts from the committed files.

| Source | Provider | Used for | Rows |
|---|---|---|---|
| NBA Stats 1947–present — *Team Summaries* | Kaggle, `sumitrodatta/nba-aba-baa-stats` | Prior-season team ratings | 1,907 |
| NBA Stats 1947–present — *Player Per Game* | Kaggle, `sumitrodatta/nba-aba-baa-stats` | Player averages, injury importance weighting | 33,339 |
| 2016–2025 NBA injury data | Kaggle, `jacquesoberweis/2016-2025-nba-injury-data` | Injury log through 2025-01-12 | 16,873 |
| NBA injury log, 2025-01-13 onward | ProSportsTransactions.com, retrieved 2026-07-28 | Injury log covering the test season | 3,581 |
| Betting odds, 2008–2026 | Supplied by Kirtan — **upstream URL not recorded** | Vegas baseline (scores stripped, see §6.4) | 24,440 |
| Game logs, 2023-24 / 2024-25 / 2025-26 | `stats.nba.com` via the `nba_api` package | Schedule, results, rest, form, head-to-head | 1,230 / 1,225 / 1,322 |
| Odds cross-check archive, 2012-13 to 2018-19 | Per-sportsbook archive carrying `Open_Line_ML` alongside Pinnacle, 5Dimes, Heritage, Bovada and BetOnline moneylines | Confirming our odds are closing lines (§6.4) | 7 seasons |

Two provenance gaps we should close before the final submission:

- **The odds file has no recorded upstream.** `data/raw/odds/primary/nba_2008-2026.csv` is
  the basis of the entire Vegas baseline, and the repository does not say where it came
  from. Kirtan supplied it and should record the URL.
- **The cross-check archive is the same.** Its column names identify it as a
  per-sportsbook line archive, but the source is not written down.

Collected but **not read** by the running system: `player_injuries_13_23` and
`player_performance` (both Kaggle, from Kirtan's 2026-07-12 candidate list). They were
evaluated and not used. We are listing them so the repository contents and the report
agree.

Two collectors exist for live data but do not feed the replay: `data/pull_games.py`
(Basketball Reference, via `basketball_reference_web_scraper`) and
`data/pull_espn_injuries.py` (the ESPN site API). `nba_api` was blocked from GitHub
Codespaces by IP, which is why Basketball Reference and ESPN were used for collection
while the committed game logs came from a local `nba_api` pull.

### 4.1 Why the injury log is two files

The Kaggle injury set stops at 2025-01-12, which covers none of the season we test on. The
continuation was pulled from ProSportsTransactions.com, the same upstream the Kaggle set
was built from, so the columns match exactly. The two files are kept separate to preserve
provenance. A test asserts they join with no gap and no overlap: 2025-01-12 is present
from the first file, 2025-01-13 from the second, and the two share no dates.

That pull is not scriptable. The site returns HTTP 403 to `curl` and to `requests`
regardless of headers, because it fingerprints the TLS handshake rather than the
User-Agent. The rows were paged out of a logged-in browser session, 25 at a time.

A better source exists and is not wired up. The `nbainjuries` package reads official NBA
injury reports and carries real filing timestamps rather than transaction dates. It needs
Java 8+ for PDF parsing, which is why we did not adopt it. See §11.3.

---

## 5. Date gating

### 5.1 The rule

Every retrieval carries an `as_of_date`. Only records published on or before that date are
returned. Anything that cannot be computed comes back null with a stated reason, never
zero. An unknown injury list is not "nobody is hurt", and a system that quietly substitutes
zero for unknown will look confident and be wrong.

### 5.2 Gate 1: the snapshot

`scripts/gate_snapshot.py` is plain Python. No model runs. It reads the data directory and
writes a filtered copy:

```
$ python -m scripts.gate_snapshot --as-of 2026-01-14

  samples/game_logs_2026.csv         1,322 kept    719 outcomes cleared
  samples/odds_only.csv             23,714 kept    726 dropped
  raw/injury_pst_2025_2026.csv       2,272 kept  1,309 dropped
  raw/Player Per Game.csv           32,606 kept    733 dropped
```

Point the agent at that directory and the future is not filtered, it is absent.

One decision inside this is not obvious. For future games the script does not drop the
row. It keeps the row and erases three columns: `home_pts`, `away_pts`, `winner`. The agent
is being asked to preview a game that has not been played, so it has to see that the game
exists. It must not see how it ended. Dropping the row would hide the question along with
the answer.

This gate exists because our advisor asked for it on 2026-07-28:

> "The first thing that happens before the agent even runs is that you run [a script] that
> copies a part of the data into a folder that has only [data] up to a certain date. And
> then you run your agent, and your agent can only look at *it*... you're not relying on
> the LLM to gate its own data. You pre-gate it."

### 5.3 Gate 2: query-time filtering

`agent/sources.py` filters every read against the as-of date independently.

The two gates are not redundant. A snapshot can only be as strict as its loosest
legitimate reader. `team_form_as_of` needs games strictly before the as-of date.
`schedule_context` needs games through it. One on-disk cut cannot satisfy both without
starving one of them. The snapshot removes what nobody may see; the filter draws the
per-tool line.

Not every date-sensitive value is a leak. The NBA publishes its full schedule in August, so
"BOS plays the 23rd and the 25th" is knowable on any date in the season, which makes rest
and back-to-back status legitimate features. The outcome of the game on the 23rd is not.
`schedule_context` makes that distinction explicitly and `models/features.py` mirrors it.

### 5.4 Gate 3: pinning the model's knowledge

Gating the data does nothing about what the language model already knows.

The scored replays run on Gemma 4, locally, through Ollama. Its knowledge cutoff is around
January 2025, which we verified behaviourally: it knows the 2024 Finals and does not know
2025 or 2026. Every game in the test window postdates that.

This has a consequence worth stating. The 2023-24 and 2024-25 seasons predate the cutoff,
so they are demonstrations of the mechanism and **not** valid evaluation games for the LLM
arms. They remain valid for training the logistic regression, which has no world knowledge
to leak. A commercial model (Claude) was used during development for iteration speed and
never for a scored run.

### 5.5 Testing the gate

A test that cannot fail proves nothing. Rather than assert that a filter was called, we
broke each rule deliberately and confirmed tests caught it.

| Mutation | Tests that failed |
|---|---|
| Feature accumulators advanced before the row was emitted | 3 |
| Form window drifted out of sync with the agent's accessor | 1 |
| Test season added to the training seasons | 3 |

The first is the important one. The natural way to compute a team's season win percentage,
grouping the season and taking the mean, silently includes the game being predicted. That
gives a very accurate and completely worthless model.

One further test spawns a real subprocess with `NBA_SNAPSHOT_DIR` set and asserts the child
resolves its data directory to the snapshot, so the in-process tests cannot pass for the
wrong reason.

The suite is 64 tests, run from a clean checkout on 2026-08-03.

---

## 6. The agent

### 6.1 The tools

The agent's entire world is seven functions. It cannot query a database, browse the web, or
invent a number. Every retrieval tool takes an as-of date.

| Tool | Status | Returns |
|---|---|---|
| `retrieve_matchup_context` | working | Ratings, rest, injuries, head-to-head as of a date |
| `retrieve_player_splits` | working | Season averages, optional back-to-back split |
| `retrieve_team_form` | working | Rolling 10-game record and point differential |
| `retrieve_injuries` | working | Who was known to be out that morning |
| `predict_win_probability` | working | Model output (withheld in arm B) |
| `retrieve_schedule` | blocked | Needs a committed forward-looking schedule table |
| `predict_stat_line` | not started | Points, rebounds, assists |

`python -m agent.run --status --source real` prints this live, so the project's blocking
list is generated from the code rather than maintained by hand.

A tool whose input does not exist returns `{"status": "awaiting_input", "needs_from": ...,
"needs": ...}` and the agent is instructed to report the gap rather than fill it. The
failure we were most concerned about is a confident, complete-looking report with an
invented number inside it.

### 6.2 Ten tools became seven

`retrieve_news` was cut because no source with reliable publication timestamps was found.
It was the highest effort of the ten and the least measurable contribution.
`predict_best_player` was cut because it depended entirely on `predict_stat_line`, which
never started. The third is not a scope cut.

### 6.3 The tool we removed

The agent had a `retrieve_betting_line` tool. Its docstring stated in capitals that the
line was context only. Running the live agent on 2026-01-14, it wrote this into its own key
factors:

> "The closing betting line favors the home team, ORL (-5.5 spread)"

The closing line is what we grade ourselves against. An agent that reads the market and
repeats it scores well and has predicted nothing.

Nothing was violated dramatically. The agent was asked what drove its reasoning and
truthfully reported a factor it had used. The problem was that the tool made the shortcut
available. Telling a model not to peek is a request; taking the tool away is a guarantee.

So the tool was removed. `agent.sources.closing_line` still exists and `eval/` still calls
it directly, so the Vegas baseline is untouched: the line is available to the scorer and
unavailable to the predictor. A test asserts the tool cannot reappear in the agent's tool
list.

### 6.4 Keeping the answer key separate

The raw odds file stores `score_away` and `score_home` in the same row as the betting line.
A tool reading that row for the spread would hand the agent the final score. This is the
most likely way the project could have leaked.

`scripts/odds_only.py` derives the odds sample through a column allowlist, so score columns
cannot reach it even if the source schema changes. That leaves two files that cannot
contaminate each other: `game_logs_2026.csv` holds schedule and results and is read only by
the eval harness after a prediction is made; `odds_only.csv` holds the market price and has
no score columns.

The Vegas comparison also rests on an assumption: that our odds are closing lines rather
than opening lines. `eval/crosscheck_odds.py` tests it against the archive in §4, which
labels the two explicitly. Nine of ten sampled games are closer to closing.

---

## 7. The model

### 7.1 What it is

A logistic regression. We chose it over a tree ensemble for three reasons: the weights are
readable in a pull request, it serialises to a few hundred bytes of named numbers rather
than a pickle, and it loads without `sklearn`, so the agent, the harness and the UI all run
without the training dependency installed.

**Input:** home team, away team, as-of date. **Output:** `home_win_prob`, a float from 0
to 1, plus provenance.

Eight features, all computed strictly from games before the one being predicted.

| Feature | Weight |
|---|---|
| `win_pct_diff` | +0.396 |
| `form_margin_diff` | +0.378 |
| `away_games_played` | −0.273 |
| `injury_weight_diff` | −0.246 |
| `home_games_played` | +0.179 |
| `home_back_to_back` | −0.147 |
| `away_back_to_back` | +0.121 |
| `rest_diff` | +0.003 |
| *(intercept)* | +0.202 |

Weights are standardised, so they are comparable to each other. The signs are all sensible:
a better record helps, a back-to-back hurts, injuries hurt, and an opponent on a
back-to-back helps.

### 7.2 The split

Trained on 2023-24 and 2024-25. Tested on 2025-26. Split by season, not by random shuffle.
A random shuffle lets the model learn from March to predict January, which inflates
accuracy by a few points that disappear under review. `TRAIN_SEASONS = (2024, 2025)` and
`TEST_SEASON = 2026` in `models/train.py`, and three tests fail if the test season enters
training.

### 7.3 Results

All 1,322 games of 2025-26:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.687 | 0.247 |
| **our model** | **66.5%** | **0.612** | **0.212** |
| Vegas closing line | 69.0% | 0.578 | 0.198 |

Train accuracy is 66.8% against test accuracy 66.5%, a gap of +0.3%. The model is not
overfit. We sit 2.5 points below the market.

We report three metrics because accuracy alone means little on NBA games. Log loss punishes
confident wrong answers. Brier is mean squared error on the probability. A system can be
accurate and badly calibrated, and for a system meant to express uncertainty that
distinction matters.

### 7.4 What the fitted model actually bought us

The hand-tuned heuristic it replaced already scored 66.3%. On raw accuracy the fitted model
gains almost nothing. It wins on calibration (log loss 0.612 against 0.617, Brier 0.212
against 0.222) and on being checkable: its split is enforced by tests, its weights are
readable, and it generalises measurably instead of being tuned by hand against the same
season it is scored on. It should not be quoted as a large accuracy jump.

---

## 8. Stat-line and win models — *shell*

> **Sarvesh** — this section is yours. It needs to answer, for each model, what the input
> is and what the output is.
>
> - The stat-line regression (points, rebounds, assists): features, output shape, split,
>   results against a baseline.
> - The XGBoost win classifier: the same. The 2026-07-28 review flagged that the classifier
>   setup may be misconfigured. Resolving that here is the most valuable thing in this
>   section.
> - Linear regression against XGBoost for the stat lines, and which won.
> - Apples-to-apples: the advisor asked that your win predictions and the agent's run on the
>   same set of games. §9 covers our 1,322. State which games yours covers.
>
> `predict_stat_line` in `agent/tools.py` is written and waiting. Drop the regression behind
> that signature and the agent picks it up with no other change. See `models/README.md`.

> **Patrick** — the data pipeline section: collection, why Basketball Reference and ESPN
> rather than `nba_api` (the Codespaces IP block), cleaning, and the rolling-5 and
> rolling-10 engineered features that feed Sarvesh's models.

> **Kirtan** — your gating function and how it relates to `scripts/gate_snapshot.py`, plus
> the odds cross-check in §6.4, which is your result. Please also record the upstream URL
> for the odds file (§4).

---

## 9. Experiment and results

### 9.1 Design

Three arms, differing by exactly one tool.

| Arm | What it is | Has `predict_win_probability`? | LLM? |
|---|---|---|---|
| A | Model only | is the model | no |
| B | Agent only | no | yes |
| C | Agent + model | yes | yes |

Arms B and C are the same agent, prompt, data and gate. The only difference is whether
`predict_win_probability` is in the tool list, controlled by one flag in `agent/tools.py`. A
test asserts the two tool lists differ by that single entry.

**Hypothesis, stated before the run: C beats both A and B.**

### 9.2 Why the LLM arms use a sample

Arms B and C call a language model once per game, about 38 seconds each. All 1,322 games
across three arms is roughly 30 hours. So B and C run on a fixed random sample of 40, and
every arm is scored on that same sample.

At n=40 the band on accuracy is about ±8%, wider than the effect we were looking for. The
harness prints one standard error next to the headline gap and refuses a full-season LLM
run rather than quietly starting a 30-hour job.

### 9.3 Scoring

The LLM does not grade itself. `eval/three_arms.py` is plain Python: it collects each arm's
probability, then compares against ground truth read from a file no tool can reach. An
unparseable answer is recorded as a skip, never coerced to 0.5, which would drag an arm
toward the baseline and flatter it.

### 9.4 Arm A, full season (n = 1,322)

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.687 | 0.247 |
| **arm A** | **66.5%** | **0.612** | **0.212** |
| Vegas | 69.0% | 0.578 | 0.198 |

### 9.5 All three arms (n = 40 each)

| arm | seed 0 | seed 1 |
|---|---|---|
| A — model only | 75.0% | 70.0% |
| B — agent only | 57.5% | 62.5% |
| C — agent + model | 55.0% | 62.5% |
| Vegas | 57.5% | 77.5% |
| always-home | 50.0% | 62.5% |

Arm A's 75% is not our accuracy. The season figure is 66.5%; these 40 games happened to
suit it. Vegas scoring 57.5% on one sample and 77.5% on the other, in the same season, shows
how noisy n=40 is. That is why the finding below is a paired one.

### 9.6 The paired finding

Comparing headline accuracies asks whether a particular 40 games flattered one arm. The
paired question avoids that: when the agent overruled the model, how often did overruling
help?

| sample | agreed | overrides | model right | agent right |
|---|---|---|---|---|
| seed 0 | 28 | 12 | 10 | 2 |
| seed 1 | 33 | 7 | 5 | 2 |
| **pooled** | **61** | **19** | **15** | **4** |

Two-sided sign test on the pooled overrides: **p ≈ 0.019**. The harness prints a one-sided
value of 0.0096. Our pre-stated hypothesis pointed the opposite way to the observed effect,
so a one-sided test in that direction would be post-hoc, and the two-sided figure is the one
we quote. The conclusion holds either way.

Seed 1 alone is not significant. Seven overrides and a 5–2 split is ordinary luck. What
replicates is the direction: the agent's overrides succeed 17% and 29% of the time, both far
below the 50% a coin would give.

The reversals were not marginal. The two largest took a confident correct call and inverted
it:

| game | model | agent | actual |
|---|---|---|---|
| CHI-ORL-2025-12-01 | 0.815 | 0.249 | home won |
| IND-PHI-2026-01-19 | 0.741 | 0.242 | home won |

---

## 10. Discussion

We predicted C > A. We got C < B < A.

The paired analysis shows this is not random noise around a good estimate. The agent is
systematically talking itself out of the model's confident correct calls. Our leading
hypothesis is that it over-weights the injury list from `retrieve_injuries`. Four names on a
list reads as decisive to a language model, while the fitted model has learned from two
seasons roughly what a given injury load is worth. The model has a calibrated prior about
injuries. The agent has a narrative one.

This is a hypothesis, not a finding. We have not yet read the per-game reasoning on the 19
overrides and categorised it. That is the top item on our open list.

The result points at a specific redesign rather than a vague conclusion: the agent should
probably annotate the model's number rather than replace it, moving it from predictor to
explainer with the model keeping the final say on the probability.

We are careful about scope. We have shown that this agent, with these seven tools, on these
79 paired games, degrades a good estimate. We have not shown that LLM agents cannot improve
on classical models in general.

A negative result is more useful here than a confirmed one would have been. A confirmed
hypothesis would have told us the architecture was fine. This tells us where it breaks, and
it came out of a design built so that a negative answer would be legible rather than
ambiguous.

---

## 11. Limitations

1. **n = 40 per LLM sample**, about a ±8% band. The paired finding pools to 19 overrides,
   enough for a sign test and not much more.
2. **`predict_stat_line` was never built.** Projected stat lines were a stated deliverable
   in the PDP. The tool signature exists and reports itself as blocked; the regression does
   not exist.
3. **Injury data are transaction dates, not news timestamps.** The log records when a player
   was placed on or activated from the injured list, not when the news broke. An as-of query
   on the morning of a game can see a same-day placement. This is a residual leak of
   unmeasured size. `nbainjuries` (§4.1) would fix it and needs Java 8+.
4. **Injury importance is a prior-season minutes and points proxy** and treats every listed
   player as fully out, so it over-penalises. Players with no prior season carry `None`
   rather than `0.0`; a rookie is unknown, not worthless.
5. **`home_games_played` and `away_games_played` carry real weight**, which is suspicious.
   The two are nearly identical for any given game, so the model may be fitting a schedule
   artifact. This deserves an ablation we did not run.
6. **No opponent-adjusted strength of schedule.** A 5–0 run against weak teams counts the
   same as 5–0 against strong ones. Probably the largest available improvement.
7. **The model-knowledge gate only holds for 2025-26.** Earlier seasons predate Gemma 4's
   cutoff.
8. **The UI's report tab is not agentic.** It runs the deterministic path. `ui/chat.py` can
   be agentic if the backend is switched. The interface says so rather than letting a demo
   imply otherwise.
9. **The Vegas baseline is in-sample in one respect.** The spread-to-probability conversion
   uses a residual sigma fitted on the same 1,322 games, which makes the baseline slightly
   stronger than it deserves. A conservative bar, but not a neutral one.
10. **Two data sources have no recorded upstream** (§4).

---

## 12. Team contributions — *shell*

> Each member completes their own row.

| Member | Lane | Delivered |
|---|---|---|
| **Josh Cannon** | Agent | The seven-tool interface, the agent loop, date-gated sources, the snapshot gate, the win-probability model, the replay and three-arm harnesses, 64 tests, both Streamlit interfaces. |
| **Patrick Haley** | Data | *(to complete — collection pipeline, cleaning, rolling-5/10 engineered features; PRs #12, #14, #15)* |
| **Sarvesh Vinod Kumar** | Models | *(to complete — linear regression and XGBoost for stat lines, XGBoost win classifier, accuracy evaluation)* |
| **Kirtan Patel** | Data / gating | *(to complete — date-gating function, candidate dataset survey, odds cross-check)* |

---

## 13. AI use disclosure

Per course policy.

This project used AI coding assistance (Claude) throughout: implementation, refactoring,
test authoring, and drafting parts of this report. We disclose it because the course
requires it and because not disclosing it would misrepresent how the work was done.

Two commitments constrain what that assistance produced. Every line merged is understood
and defensible by its author; the architectural decisions in this report were ours, and each
has a reason behind it. And no result here was produced by an LLM judging its own output:
every number came from Python scoring against ground truth in a file no agent tool can
reach, regenerated from a clean checkout on 2026-08-03.

The project's main finding is itself a caution about uncritical AI use. An agent given a
tool it was told not to lean on leaned on it, and an agent given a good estimate made it
worse.

---

## Appendix A — Reproducing the numbers

From a clean clone, Python 3.11+:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                  # 64 tests (§5.5)
python -m models.train                  # §7.1 weights, §7.3 accuracy
python -m eval.three_arms               # §9.4
python eval/crosscheck_odds.py          # §6.4
python -m scripts.gate_snapshot --as-of 2026-01-14   # §5.2
python -m agent.run --status --source real           # §6.1
```

§9.5 and §9.6 come from `eval/results_three_arms_sample40.csv` and `..._seed1.csv`. To
regenerate (needs `ollama serve` and `ollama pull gemma4`, about 50 minutes each):

```bash
python -m eval.three_arms --arms abc --sample 40 --model ollama
python -m eval.three_arms --arms abc --sample 40 --seed 1 --model ollama
```

Interface:

```bash
streamlit run ui/app.py     # pregame report, tools, gating proof, build status
streamlit run ui/chat.py    # conversational view over the same tools
```

## Appendix B — References

**Data.** Full provenance and row counts are in §4.

- Datta, S. *NBA/ABA/BAA Stats (1947–present).* Kaggle. `kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats`
- Oberweis, J. *2016–2025 NBA Injury Data.* Kaggle. `kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data`
- ProSportsTransactions.com, basketball transaction search, injury filter. Retrieved 2026-07-28.
- NBA Stats API (`stats.nba.com`), accessed via the `nba_api` Python package.
- ESPN NBA injuries API, `site.api.espn.com`.
- Basketball Reference, accessed via `basketball_reference_web_scraper`.
- Betting odds 2008–2026 and the per-sportsbook cross-check archive — *upstream not yet recorded; see §4.*

**Software.**

- LangChain 1.x — agent loop and tool interface.
- Ollama with Gemma 4 — the local, cutoff-pinned model used for scored replays.
- Anthropic Claude — development iteration only, never a scored run.
- scikit-learn — model fitting. pandas — data preparation. Streamlit — interface. pytest — tests.

**Course sources.**

- Advisor meeting, 2026-07-28 (pre-gating architecture, independent evaluation, apples-to-apples game set, report structure).
- Advisor meeting, 2026-07-21 (betting line as evaluation baseline rather than model input).
- PDP review, 2026-07-07 (date-gated retrieval architecture, role split).
- Kirtan Patel, candidate datasets email, 2026-07-12.
