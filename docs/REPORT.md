# Can an AI Agent Predict NBA Games? Three Approaches, Compared

**CECS 499: Senior Transdisciplinary Capstone, Summer 2026**

**University of Tennessee, Knoxville**

Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel

Advisor: Prof. Amir Sadovnik

Repository: `github.com/joshcannonai/nba-game-intelligence-agent`

---

> **Draft, 2026-08-03.** The models in §5.2 and the per-member breakdown in §6 are
> marked for their owners. Every number came from running the code on 2026-08-03;
> Appendix A lists the command for each one.

---

### What we call things

Three different things here could be called "the model", so we don't call any of them
that.

| Term | What it means |
|---|---|
| **the predictor** | A small statistical model we trained. Two teams and a date in, a percentage chance the home team wins out. No AI. |
| **the agent** | An AI chatbot given seven functions it can call to look things up. It reads the results and writes a report. |
| **a tool** | One of those seven functions. The agent can *only* use these. No web, no database. |
| **the closing line** | What sportsbooks think will happen, as odds. Our benchmark, because it's hard to beat. |

---

## 1. Abstract

We built a system that predicts who wins NBA games, then tested three ways of making
that prediction to see which works best.

The first is a **predictor**, a statistical model trained on two past seasons. The
second is an **agent**, an AI chatbot with seven lookup tools that reasons its way to an
answer. The third gives the agent the predictor's number and lets it decide whether to
agree.

We expected the third to win, since it knows the most. It came last. When the agent
disagreed with the predictor, it was wrong 15 times out of 19.

All of it rests on one rule: nothing the system sees may be dated after the game it's
predicting. That sounds obvious and is easy to get wrong, especially with an AI that may
simply remember who won.

We also converted the results into money. Betting $100 a game on the predictor's picks
across a season loses **$2,135**. That is a smaller loss than always backing the favourite,
which loses $4,628 while winning *more* games.

## 2. Introduction

Predicting basketball is not new. Sportsbooks do it for money and they're good at it. We
were never going to beat them, and that wasn't the goal.

The goal was to find out what we could build, then measure honestly which parts helped.
Specifically: does adding an AI agent to a statistical prediction make it better, worse,
or no different?

**Why it's worth doing.** AI agents are being bolted onto existing systems everywhere,
usually assuming that helps. Sometimes it does. We had a setup where we could actually
check (same data, same games, one thing changing), and that method transfers to anyone
making the same call.

**The hard part.** Testing a normal model is routine: hide a season during training, then
see how it does. But an AI chatbot has read most of the internet. Ask it about a game
from December 2025 and it might just remember the score.

A system that remembers looks *fantastic*: near-perfect scores, a demo that lands, and
no actual prediction happening. So a lot of this project is plumbing that stops the
system reaching the answer. Without it, none of the numbers mean anything.

## 3. Previous Work

**Public NBA forecasting.** FiveThirtyEight published NBA predictions for years, first
with an Elo rating system (a chess-style rating where teams gain and lose points on
results) and later a player-based model called RAPTOR. Both are public. They had a
full-time team; we had a semester. We treat their work as a sanity check on what's
realistic, not a target.

**Betting markets as a benchmark.** In sports economics the closing line is generally
treated as a strong summary of everything known about a matchup. That's why we use it as
our benchmark rather than an input. That distinction mattered in practice, not just in
theory (§5.4).

**AI agents that use tools.** Our agent alternates between thinking and calling a
function, then reads the result before deciding what to do next. That is the pattern popularised
as ReAct (Yao et al., 2023), implemented with LangChain. Nothing about the loop is novel.
What we changed is what the agent is *allowed* to reach.

**Testing AI on data it may have memorised.** A known problem in AI evaluation, which is
why benchmark scores increasingly get published alongside a model's knowledge cutoff. Our
fix is simple: use a model whose cutoff we can check, and test only on games after it.

**How ours differs.** NBA prediction work tries to be accurate and reports accuracy. Agent
research demonstrates what agents can do. We hold everything constant and vary only
whether an AI is involved. What changed was that it got worse, which is not a result either
field looks for.

> **Team note:** this cites what we can stand behind. If anyone has read a specific paper
> on NBA prediction or market efficiency, add it. It's worth 10% and it's thin.

## 4. Dataset

Six sources, all committed so the project runs from a fresh download.

| Source | Where from | Used for | Size |
|---|---|---|---|
| NBA team stats, 1947–present | Kaggle | Last season's team ratings | 1,907 rows |
| NBA player stats, 1947–present | Kaggle | Player averages, injury weighting | 33,339 rows |
| NBA injuries, 2016–2025 | Kaggle | Who was hurt, to Jan 2025 | 16,873 rows |
| NBA injuries, 2025 onward | ProSportsTransactions.com | Who was hurt in our test season | 3,581 rows |
| Betting odds, 2008–2026 | *source not recorded* | Our benchmark | 24,440 rows |
| Schedules and results | NBA stats site | Who played who, and who won | ~1,300/season |

**Four problems we had to fix.**

*The injury data stopped too early.* The Kaggle set ends January 2025, before our test
season starts. We extended it from ProSportsTransactions.com, where that Kaggle set
originally came from, so the columns line up. That site blocks automated downloads, so
rows were collected by hand through a browser, 25 at a time. The two files stay separate
so it's clear which rows came from where, and a test checks they join with no gap and no
double-counting.

*The odds file contains the answer.* Each row has the betting line and the final score
side by side, so a tool fetching the line would hand the agent the result. We built a
filtered copy keeping only an allowed list of columns, so scores can't get through even if
the original changes.

*Season totals mislead mid-season.* The Kaggle stats are end-of-season numbers, so using
them for a December game means using March data. We use them only for "last completed
season" comparisons; current form comes from game-by-game records.

*Two sources have no recorded origin.* The odds file and the file we verified it against
arrived without a link. Our entire benchmark rests on that odds file, so we're flagging it
rather than quietly moving on. We did confirm it holds *closing* rather than opening odds
by checking ten games against an independent source: nine matched closing.

> **Patrick:** the collection pipeline and cleaning steps belong here. This section is
> worth 15% and is currently the thinnest in the report.

## 5. Technical Approach

### 5.1 Keeping the future out

Every lookup carries a date. Only information published on or before it comes back. If
something can't be worked out the answer is "unknown" with a reason, never zero. An empty
injury list because nobody's hurt and one because we don't know are very different, and
treating the second as the first produces a confident wrong answer.

We block the future twice. First, a plain script copies the data into a fresh folder with
everything after the chosen date stripped out. No AI involved. Point the agent at that
folder and the future isn't filtered. It isn't there. Second, each tool applies its own
date check on top.

That isn't belt-and-braces: the tools need genuinely different cutoffs. Recent-form needs
games strictly *before* the date; schedule lookup needs games *through* it. One folder
can't satisfy both.

One thing that is *not* cheating: knowing when games are scheduled. The NBA publishes its
calendar in August, so "this team played last night" is fair on any date. Only *results*
are hidden.

Separately, we had to handle what the AI already knows. Blocking data doesn't help if the
chatbot remembers the score, so scored runs use a model that runs on a laptop with a
knowledge cutoff around January 2025, before every game we test on.

### 5.2 The predictor

A **logistic regression**: it takes eight numbers about a matchup, multiplies each by a
weight, adds them up, and squashes the total into a percentage.

We picked it over fancier options on purpose. You can read its weights and argue with
them. It saves as a few hundred bytes of plain text. It runs anywhere.

The eight inputs are all *differences* between the two teams: win percentage, average
scoring margin over ten games, rest days, whether either played yesterday, injury load,
and games played. Each is calculated only from games before the one being predicted.

That last point is where projects like this usually break. The obvious way to calculate a
team's win percentage is their season record, but that includes the game you're
predicting. You get a model that looks astonishing and is worthless. Our code updates
every running total *after* recording each game.

We also split by **season**, not randomly. A random split lets the model learn from March
to predict January, inflating the score by points that vanish under review.

> **Sarvesh:** your two models go here. For each, say plainly what goes in and what comes
> out, how you split train and test, and what you compared against. The 07-28 review
> flagged the win classifier might be set up as a regressor. Sorting that out is the most
> useful thing in this section.

### 5.3 How we score it

**Accuracy** is how often the pick was right. **Log loss** punishes being confident *and*
wrong. **Brier score** measures how far the predicted percentage sat from what actually
happened. Two reference points: always picking home (they win ~55%), and the closing line.

### 5.4 The three approaches

- **A: the predictor alone.** No AI.
- **B: the agent alone.** Lookup tools but no predictor; it reasons to its own number.
- **C: the agent plus the predictor.** Same as B, plus the predictor's answer.

B and C are the same AI, same instructions, same data. The only difference is whether the
predictor's number is available, and a test enforces that.

**One tool we removed.** The agent originally had a betting-line lookup. Its instructions
said in capitals that the line was background only. Running it live, we watched it write
*"The closing betting line favors the home team, ORL (-5.5 spread)"* into its own
reasoning.

The betting line is what we grade it against. An agent that reads the line and repeats it
scores brilliantly and has predicted nothing. Nothing dramatic happened. It was asked
what informed its answer and said honestly. The tool just made it possible. Telling a
model not to use something is a request; taking the tool away is a guarantee.

## 6. Implementation

Python 3.11+. The agent uses **LangChain**; the predictor trains with **scikit-learn** and
saves as plain JSON; data work uses **pandas**; the interface is **Streamlit**; tests use
**pytest**. Scored agent runs use **Gemma 4** through **Ollama**, locally, with no API key
and no cost.

The repository splits by job: `agent/` (tools and loop), `models/` (the predictor),
`eval/` (scoring), `scripts/` (data prep and the date filter), `skills/`, `ui/`, `tests/`.

**The seven tools** are the agent's whole world: matchup context, team form, injuries,
player splits, the predictor's number, the schedule, and projected player stats. All seven
return real data. `python -m agent.run --status` prints what works, generated from the code
rather than a list we maintain by hand.

**Skills.** Each tool has a plain-English rules file saying when to use it and how to read
the result. These load into the agent's instructions at startup, so a teammate can change
the agent's behaviour by editing a text file.

**Why not a database.** We considered letting the agent query a database freely and didn't,
for the same reason we removed the betting-line tool. An open query surface is how a system
reaches data nobody meant it to have.

**Testing.** 95 tests. Rather than just check a filter runs, we deliberately broke each
rule and confirmed the tests noticed: feeding future data in early (3 tests caught it),
letting two parts of the code drift apart (1), training on the test season (3). Another
launches a separate process to confirm the agent really reads the filtered folder.

> **Patrick / Sarvesh:** libraries and tooling specific to your parts belong here too.

## 7. Results

### 7.1 Accuracy

The predictor on all 1,322 games of 2025-26, which it never trained on:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.687 | 0.247 |
| **the predictor** | **66.5%** | **0.612** | **0.212** |
| closing line | 69.0% | 0.578 | 0.198 |

It scored 66.8% on training data and 66.5% on unseen data. That 0.3% gap means it learned
patterns rather than memorising games. It sits 2.5 points behind the sportsbooks.

Worth being honest about: a hand-tuned rule of thumb we wrote earlier already scored 66.3%.
On raw accuracy the trained predictor barely improves on it. What it does better is express
confidence sensibly, and it can be checked by tests rather than tuned by feel.

### 7.2 All three approaches

The AI approaches take ~40 seconds per game, so they ran on two 40-game samples, with every
approach scored on the same games.

| | A: predictor | B: agent alone | C: agent + predictor |
|---|---|---|---|
| sample 1 | 75.0% | 57.5% | 55.0% |
| sample 2 | 70.0% | 62.5% | 62.5% |

**C came last.** Since 40-game accuracy is noisy, the better question is: where the agent
*disagreed* with the predictor, who was right? Pooling both samples, it disagreed 19 times
and was wrong on 15.

We then gave each tool written rules, including not to second-guess the predictor on
injuries, and re-ran the same games. C's accuracy went from 58.8% to **66.3%** and
disagreements dropped from 19 to 11. Roughly half the gap closed, on both samples. That's
encouraging rather than settled: 11 disagreements isn't enough to call it solid.

### 7.3 What it's worth in money

Accuracy is abstract, so: **bet $100 on whoever each approach picked. Who ends up ahead?**

Our odds file has no moneyline prices for this season, only spreads, so prices had to be
reconstructed. The scores and spreads are real; the prices are calculated, and we tested
that calculation against the **19,807 earlier games in the same file that do carry real
quoted prices**:

| | |
|---|---|
| correlation with real prices | **0.9959** |
| average error | 2.9 percentage points |
| within 5 points | 77.6% of games |

The house margin is measured from those same games (3.75%), not assumed.

**Full season, 1,322 games:**

| approach | games won | profit | return |
|---|---|---|---|
| **the predictor** | 66.5% | **−$2,135** | −1.6% |
| always bet the favourite | 69.0% | −$4,628 | −3.5% |
| always bet the home team | 55.5% | −$7,350 | −5.6% |

Everything loses money. That's the correct result and the most useful number here. The
house margin is the bar and none of our approaches clears it.

But look at the top two rows. **Always backing the favourite wins more games (69.0% vs
66.5%) and loses nearly twice as much money.** Favourites win often and pay badly. Our
predictor picks more selectively, so it loses less. That gap is the whole argument for why
accuracy alone is a poor way to judge a prediction system.

On the 80 games where all three ran, A made +$993 and C went from −$832 to +$77 once
constrained. **Do not read A's +$993 as a profitable system**. The same predictor loses
$2,135 across the full season. Those 80 games flattered it, which is exactly the trap this
report is about, and we'd rather demonstrate it on ourselves than claim an edge.

### 7.4 A measurement that changed our design

We intended to write a rule like "star player out, drop the prediction by N%". No N is
supported by the data.

| comparison | result |
|---|---|
| all games | teams win *more* without their star (clearly wrong) |
| only teams that have a star | +5.6% more wins without him |
| **each team against itself** | **+0.0%** |

The first two are distorted the same way: having a 20-point-per-game scorer is a property
of *good teams*, and a good team is still good on the night that player sits. Comparing a
team only against itself removes that, and the effect vanishes.

## 8. Conclusion and Future Work

We built a working NBA prediction system, tested three ways of making the prediction, and
measured which was best. The statistical predictor won. Adding an AI agent made it worse,
mostly by talking itself out of correct answers, and giving that agent explicit written
rules recovered about half the damage.

None of the approaches makes money against realistic prices. The predictor loses least,
and notably loses less than a strategy that wins more games.

**What we'd tell someone repeating this.** Verify your AI hasn't seen the test data rather
than assuming. Change one thing at a time. Convert accuracy into whatever unit actually
matters, because a system can be more accurate and still worse. And distrust small samples,
including flattering ones.

**Limitations.** The AI comparisons rest on 80 games. Our injury data records when players
were placed on the injured list, not when the news broke, so a same-day entry can slip in.
We link players to teams using last season's rosters, so "injured" and "changed teams"
aren't cleanly separated. Two data sources have no recorded origin. And projected player
statistics, promised in the proposal, were never built.

**Future work.** The immediate step is reading the agent's own reasoning on the 19
disagreements to test our theory that it over-weights injuries. Beyond that: adjusting for
strength of schedule, the biggest single improvement available; using official timestamped
injury reports instead of transaction records; running the AI comparisons on far more
games; and the experiment our advisor suggested, deliberately letting the agent see the
future, to measure what cheating is worth.

## 9. References

1. Datta, S. *NBA/ABA/BAA Stats (1947–present)*. Kaggle.
2. Oberweis, J. *2016–2025 NBA Injury Data*. Kaggle.
3. ProSportsTransactions.com, basketball transactions archive. Retrieved 2026-07-28.
4. NBA Stats API (`stats.nba.com`), via the `nba_api` package.
5. Basketball Reference, via `basketball_reference_web_scraper`.
6. ESPN NBA injuries API.
7. Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
8. FiveThirtyEight NBA forecasts. Elo and RAPTOR methodology.
9. LangChain, agent framework. Ollama / Gemma 4, local AI model.
10. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12 (2011).

> **Team note:** 7, 8 and 10 are real and correctly attributed, but nobody has read 7 or 10
> properly. Either read them or drop them. A citation you can't discuss is worse than a
> shorter list.

---

## Appendix A: Reproducing the numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                 # 95 tests
python -m models.train                 # §5.2 weights, §7.1 accuracy
python -m eval.three_arms              # §7.1 predictor vs baselines
python -m eval.betting                 # §7.3 the money
python -m eval.betting --validate      # §7.3 price check vs real quoted lines
python -m eval.injury_impact           # §7.4 the star-player measurement
python eval/crosscheck_odds.py         # §4 closing-vs-opening check
python -m scripts.gate_snapshot --as-of 2026-01-14   # §5.1 the date filter
python -m agent.run --status --source real           # §6 what works

# the agent itself, locally (needs `ollama serve`, `ollama pull gemma4`)
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```
