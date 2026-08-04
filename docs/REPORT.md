# Predicting NBA Games Without Seeing the Future: An Agent, a Model, and a Leakage-Controlled Comparison

**CECS 499 — Senior Transdisciplinary Capstone, Summer 2026**

**University of Tennessee, Knoxville**

Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel

Advisor: Prof. Amir Sadovnik

Repository: `github.com/joshcannonai/nba-game-intelligence-agent`

---

> **Draft, 2026-08-03.** Sections 1–8 are written. Section 5.2 (Sarvesh's models) and the
> per-member detail in Section 6 are marked for their owners. Every number was regenerated
> from a clean checkout on 2026-08-03; Appendix A lists the command that produces each one.

---

## 1. Abstract

We built a system that predicts the outcome of NBA games and explains its reasoning, and
then ran a controlled experiment to test whether the explaining part helps. The system has
two halves that answer the same question by different means: a logistic regression over
eight engineered features, and a large language model agent that calls seven retrieval
tools and writes a pregame report. Both are constrained by a single rule — every query
carries an *as-of date*, and nothing published after that date can reach either one.

What separates this from existing NBA prediction work is not the prediction. Public
forecasting systems already beat what a semester project can build, and the betting market
beats those. Our contribution is the **evaluation design**: a way to test a language-model
system on a season that has already happened without the model simply remembering the
answer. On 1,322 held-out games the model scores 66.5% against 55.5% for always picking
the home team and 69.0% for the closing line. Our stated hypothesis — that an agent given
the model's number would beat the model alone — was wrong, and measurably so.

## 2. Introduction

Anyone can predict NBA games badly. The interesting problem is knowing whether you have
predicted them well, and for a system built on a large language model that is genuinely
hard.

The difficulty is that the model has read the internet. Ask a modern LLM who won a game in
December 2025 and it may simply know. Any accuracy figure produced that way measures
memory, not prediction — and it will look excellent, which is what makes it dangerous. A
system can appear to work, pass a demo, and be measuring nothing.

This is a specific instance of a general problem in applied machine learning: the
evaluation is easier to get wrong than the model, and a broken evaluation fails silently.
It is also, as our advisor framed it during the semester, an alignment problem. A system
optimising for a goal takes whatever route reaches it, and the shortest route is often the
one a person would call cheating. The system has no concept of cheating. Telling it not to
cheat in a prompt does not work; the shortcut has to be made unavailable.

So the project's real subject is **leakage control**, and NBA prediction is the vehicle for
demonstrating it. That framing decides what is worth building. We spent our effort on the
gating architecture, the experimental design, and the tests that prove the gate holds,
rather than on chasing accuracy we were never going to win.

Who this is useful for: anyone evaluating an LLM system against historical data, which is
a large and growing category. The pattern generalises well beyond basketball — a
pre-filtered data snapshot, tools that carry an as-of date, an independent scorer, and a
model whose knowledge cutoff is verifiably older than the test window. The NBA is a
convenient testbed because outcomes are unambiguous, the schedule is public in advance,
and a strong external benchmark exists in the betting market.

## 3. Previous Work

**Public NBA forecasting.** FiveThirtyEight published NBA game forecasts for years, first
using an Elo rating system and later the RAPTOR player-rating model, with methodology and
historical predictions released publicly. These set the reference for what a well-resourced
statistical approach achieves, and they are the closest public analogue to our arm A. Our
model is far simpler — eight features against their full player-tracking pipeline — so we
treat their existence as calibration on what a semester project should expect, not as
something to beat.

**The betting market as a benchmark.** Sports economics has long treated closing lines as
a strong aggregator of available information, which is why we use the closing line as our
upper reference rather than as a feature. This distinction turned out to matter
operationally, not just methodologically (Section 5.4).

**Tool-using language model agents.** Our agent follows the now-standard pattern of
interleaving reasoning with tool calls, in the lineage of the ReAct framework (Yao et al.,
2023), implemented via LangChain's agent interface. The novel part in our setting is not
the loop but the constraint: every tool is date-gated, and one tool was deliberately
removed after it was observed leaking a benchmark.

**Evaluation contamination.** The problem of test data appearing in a language model's
training corpus is well recognised in the LLM evaluation literature and is why benchmark
scores are increasingly reported alongside cutoff dates. Our contribution is a concrete
mitigation for a time-series setting: pin the model to a verifiable cutoff and test only on
events that postdate it.

**How this differs.** Existing NBA prediction work optimises accuracy and reports it.
Existing agent work demonstrates capability on tasks where contamination is not a concern.
We are doing neither: we hold the data constant, vary only whether a language model is in
the loop, and report what that changes. We found that it made things worse, which is not a
result the prediction literature is set up to surface.

> **Team note:** this section currently cites what we can stand behind. If anyone has read
> a specific paper on NBA outcome prediction or sports betting market efficiency, add it
> with a full citation rather than leaving this thinner than the 10% it is worth.

## 4. Dataset

We use six sources. All are committed to the repository so the project is reproducible from
a clean clone.

| Source | Provider | Used for | Rows |
|---|---|---|---|
| NBA Stats 1947–present, *Team Summaries* | Kaggle (`sumitrodatta/nba-aba-baa-stats`) | Prior-season team ratings | 1,907 |
| NBA Stats 1947–present, *Player Per Game* | Kaggle (same set) | Player averages, injury weighting | 33,339 |
| 2016–2025 NBA injury data | Kaggle (`jacquesoberweis/…`) | Injury log to 2025-01-12 | 16,873 |
| NBA injury log, 2025-01-13 on | ProSportsTransactions.com | Injury log for the test season | 3,581 |
| Betting odds 2008–2026 | *upstream not recorded* | Vegas benchmark | 24,440 |
| Game logs, three seasons | `stats.nba.com` via `nba_api` | Schedule, results, rest, form | 1,230 / 1,225 / 1,322 |

**Collection.** Game logs were pulled with the `nba_api` package. Note that `nba_api` is
blocked from GitHub Codespaces by IP, which we confirmed by timeout testing, so collection
scripts also exist against Basketball Reference (`basketball_reference_web_scraper`) and
the ESPN injuries API for live use.

**Four problems we had to fix.**

*The injury log ended before our test season.* The Kaggle set stops at 2025-01-12, covering
none of 2025-26. We extended it from ProSportsTransactions.com, the same upstream the
Kaggle set derives from, so the columns match. That pull is not scriptable — the site
returns HTTP 403 to `curl` and `requests` regardless of headers because it fingerprints the
TLS handshake — so rows were paged out of a browser session. The two files are kept
separate to preserve provenance, and a test asserts they join with no gap and no overlap.

*The odds file contains the answer.* The raw odds source stores `score_away` and
`score_home` in the same row as the betting line. A retrieval tool reading that row for the
spread would hand the agent the final score. We derive the odds sample through a column
allowlist so score columns cannot reach it even if the upstream schema changes.

*The season aggregates leak if used naively.* The Kaggle team and player tables are
end-of-season totals. Using them mid-season means a December prediction is informed by
March. They are used only for prior-completed-season fallbacks; current strength comes from
game logs.

*Two sources have no recorded provenance.* The odds file and its cross-check archive were
supplied without a URL. Since the entire benchmark rests on the odds file, this is a real
gap and is flagged rather than hidden. We did verify the file contains *closing* rather
than *opening* lines by comparing against an independent per-sportsbook archive: 9 of 10
sampled games are closer to closing.

Two further Kaggle sets were evaluated and not used.

## 5. Technical Approach

### 5.1 Date-gated retrieval

The central mechanism is that every data access carries an as-of date **D**, and only
records published on or before **D** may be returned. Anything not computable returns null
with a stated reason, never zero — an unknown injury list is not "nobody is hurt", and
silently substituting zero produces a confident wrong answer.

The gate runs at two levels, which are complementary rather than redundant. A **snapshot**
step copies the data directory into a filtered copy containing nothing after **D**; a
**query-time filter** then applies per-tool rules on top. Both are needed because a
snapshot can only be as strict as its loosest legitimate reader: rolling form needs games
strictly *before* **D**, while schedule context needs games *through* it. One on-disk cut
cannot satisfy both without starving one.

An important asymmetry: not every date-sensitive value is a leak. The NBA publishes its
schedule in August, so rest and back-to-back status are knowable on any in-season date and
are legitimate features. Only *outcomes* are gated.

A third gate addresses the model itself. Gating data does nothing about what a language
model already knows, so scored runs use a local model whose knowledge cutoff (≈January
2025) verifiably predates the test window.

### 5.2 The prediction model

We use **logistic regression**: the probability of a home win is modelled as
σ(w·x + b), where σ is the logistic function, x is a standardised eight-feature vector, and
w is fitted by minimising log loss over the training seasons.

We chose it over a gradient-boosted ensemble deliberately. Its weights are directly
interpretable and comparable once features are standardised, which means the model can be
argued with in review rather than merely accepted; it serialises to a few hundred bytes of
named numbers rather than an opaque binary; and it loads without the training dependency,
so every downstream consumer runs without `sklearn`.

The features are differences between the two teams: win percentage, rolling ten-game point
margin, rest days, two back-to-back indicators, injury load, and games played. Every value
is computed strictly from games *preceding* the one being predicted.

The critical subtlety is in feature construction. The natural way to compute a season win
percentage — group the season and take the mean — silently includes the game being
predicted. That produces a highly accurate and completely worthless model. Every
accumulator is therefore advanced *after* emitting each row.

We split by **season**, not by random shuffle. A random shuffle permits learning from March
to predict January, which inflates accuracy by a few points that vanish under scrutiny.

> **Sarvesh:** your models belong here — the stat-line regression and the XGBoost
> classifier. For each, state the input and the output explicitly, the split, and the
> comparison against a baseline. The 07-28 review flagged that the classifier may be
> configured as a regressor; resolving that is the most valuable thing in this section.

### 5.3 Evaluation metrics

We report three, because accuracy alone is close to meaningless on NBA games. **Accuracy**
is the share of games called correctly. **Log loss** punishes confident wrong answers.
**Brier score** is mean squared error on the probability. A system can be accurate and
badly calibrated, and for a system expressing uncertainty that distinction matters. Two
reference points bound the problem: always picking the home team (~55%) and the de-vigged
closing line.

### 5.4 The three-arm design

Three ways of answering the same question, differing by exactly one tool:

- **A** — model only, no language model involved.
- **B** — agent only, reasoning from the retrieval tools.
- **C** — agent plus the model's number.

Arms B and C use the same agent, prompt, data and gate; the only difference is whether the
win-probability tool is available. The difference *is* the measurement, and a test enforces
that the tool lists differ by exactly one entry.

**Benchmark leakage.** Running the agent live, we observed it write *"The closing betting
line favors the home team, ORL (-5.5 spread)"* into its own reasoning. The closing line is
what we grade against; an agent that reads the market and repeats it scores well and has
predicted nothing. The tool's documentation already said it was context only. Instructing a
model not to use something is a request; removing the tool is a guarantee. We removed it,
and a test prevents its return.

## 6. Implementation

Python 3.11+. The agent loop uses **LangChain 1.x**; the model is fitted with
**scikit-learn** and served from committed JSON; data preparation uses **pandas** and the
standard library `csv`; the interface is **Streamlit**; tests use **pytest**. Scored agent
runs use **Gemma 4 via Ollama**, locally and without an API key.

The repository is organised by lane: `agent/` (tools, loop, gated sources), `models/`
(features, training, prediction), `eval/` (harnesses), `scripts/` (snapshot gate, dataset
construction), `skills/`, `ui/`, and `tests/`.

**Tools.** The agent's entire capability is seven functions — matchup context, player
splits, team form, injuries, win probability, schedule, and stat line. It cannot query a
database, browse, or invent a number. Two of the seven return `awaiting_input` because
their inputs were never built; the agent reports these gaps rather than filling them, and
`python -m agent.run --status` prints the project's blocking list generated from the code.

**Skills.** Each tool has a Markdown rule file specifying when to call it and how to read
the result. These load into the system prompt at startup, so domain rules are editable by
teammates without touching Python. The block is assembled from the tools actually granted,
preserving the one-variable difference between arms.

**Why not a database.** We considered loading everything into Postgres and letting the
agent query freely. We rejected it for the same reason the betting-line tool was removed: a
free-form query surface is how an agent reaches data nobody intended it to see, and the
gate only holds because every read passes through one module.

**Testing.** 73 tests. Rather than assert a filter was called, we broke each rule
deliberately and confirmed tests caught it — advancing accumulators early (3 tests caught
it), drifting the form window (1), and training on the test season (3). One test spawns a
real subprocess to confirm the agent reads the snapshot rather than the repository.

## 7. Results

**Arm A, all 1,322 games of 2025-26** (trained on 2023-24 and 2024-25):

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.687 | 0.247 |
| **our model** | **66.5%** | **0.612** | **0.212** |
| closing line | 69.0% | 0.578 | 0.198 |

Train accuracy is 66.8% against test 66.5% — a generalisation gap of +0.3%, so the model is
not overfit. We sit 2.5 points below the market.

An honest note on what fitting bought us: the hand-tuned heuristic it replaced already
scored 66.3%. On raw accuracy the fitted model gains almost nothing. It wins on calibration
and on being checkable.

**The three arms.** Arms B and C call a language model once per game (~40s), so they run on
fixed 40-game samples with every arm scored on the same games.

| | arm A | arm B | arm C |
|---|---|---|---|
| seed 0 | 75.0% | 57.5% | 55.0% |
| seed 1 | 70.0% | 62.5% | 62.5% |

**Our hypothesis was wrong.** C did not beat A; it lost badly. Because headline accuracies
on 40 games are noisy, the robust measurement is paired: *when the agent overruled the
model, how often did overruling help?* Pooled across both samples, the agent overruled 19
times and was **wrong on 15 of them** (two-sided sign test, p ≈ 0.019).

**Acting on the finding.** We gave each tool written rules, including an explicit
instruction not to re-price injuries on top of the model, and re-ran on the same games:

| pooled (n=80) | before | after |
|---|---|---|
| arm C accuracy | 58.8% | **66.3%** |
| log loss | 0.674 | **0.591** |
| overrides | 19 | **11** |

Arm C recovered about half the gap to arm A, replicating on both seeds by the mechanism
predicted. This is directional rather than established: the override sign test is now
*under-powered* (11 overrides, p ≈ 0.23) rather than passed.

**A measurement that changed our design.** We intended to encode "a >20 ppg player is out,
so reduce the probability by N%". No N is supportable. Comparing every game suggests teams
win *more* without their star; restricting to teams that have one gives +5.6% (z = 2.6),
still backwards. Both are confounded — having a star is a property of good teams. Comparing
each team **against itself** gives **+0.0% (se 3.3%, n = 21 teams)**, with a spread from
−32% to +36%.

## 8. Conclusion and Future Work

We set out to build an agent that predicts NBA games and explains itself, and we built a
method for finding out whether the explanation costs anything. It does. The agent
systematically talked itself out of the model's confident correct calls, most likely by
over-weighting the injury list, and constraining it with written rules recovered about half
of what it was losing.

Three things we would defend as contributions. The **two-layer gate plus a cutoff-pinned
model** is a reusable pattern for evaluating LLM systems on historical data. The **paired
override analysis** extracts a usable signal from samples far too small for headline
comparison. And the **benchmark-leakage incident** is a concrete case of an agent
optimising into the answer key, fixed by removing capability rather than adding
instruction.

The scope of our claim is narrow: *this* agent, with *these* seven tools, on *these* 80
paired games, degraded a good estimate. We have not shown that LLM agents cannot help.

**Limitations.** Sample sizes for the LLM arms are small. Injury records are transaction
dates rather than news timestamps, so a same-day placement can appear. Star-to-team mapping
comes from the prior season and cannot fully separate "injured" from "changed teams". Two
data sources lack recorded provenance. `predict_stat_line`, a stated deliverable, was never
built.

**Future work.** The immediate next step is reading the agent's per-game reasoning on the
19 overrides to test the injury hypothesis directly. Beyond that: opponent-adjusted
strength of schedule, which is the largest single modelling gain available; official
timestamped injury reports via the `nbainjuries` package; larger samples for the LLM arms;
and the ablation our advisor suggested — deliberately un-gating the agent to measure what
leakage is worth.

## 9. References

1. Datta, S. *NBA/ABA/BAA Stats (1947–present)*. Kaggle. `kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats`
2. Oberweis, J. *2016–2025 NBA Injury Data*. Kaggle. `kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data`
3. ProSportsTransactions.com — basketball transaction archive, injury filter. Retrieved 2026-07-28.
4. NBA Stats API (`stats.nba.com`), accessed via the `nba_api` Python package.
5. Basketball Reference, accessed via `basketball_reference_web_scraper`.
6. ESPN NBA injuries API, `site.api.espn.com`.
7. Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
8. FiveThirtyEight NBA forecasts — Elo and RAPTOR methodology and published historical predictions.
9. LangChain (1.x) — agent and tool interface. Ollama / Gemma 4 — local inference.
10. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12 (2011).

> **Team note:** references 7, 8 and 10 are real and correctly attributed, but nobody has
> read 7 or 10 end to end. Either read them or drop them — a citation we cannot discuss is
> worse than a shorter list.

---

## Appendix A — Reproducing the numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                 # 73 tests
python -m models.train                 # §5.2 weights, §7 accuracy
python -m eval.three_arms              # §7 arm A and baselines
python -m eval.injury_impact           # §7 the top-scorer measurement
python eval/crosscheck_odds.py         # §4 closing-vs-opening check
python -m scripts.gate_snapshot --as-of 2026-01-14    # §5.1
python -m agent.run --status --source real            # §6

# the local agent end to end (needs `ollama serve`, `ollama pull gemma4`)
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```
