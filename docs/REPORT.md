# Can an AI Agent Predict NBA Games? Three Approaches, Compared

**CECS 499 — Senior Transdisciplinary Capstone, Summer 2026**

**University of Tennessee, Knoxville**

Josh Cannon · Patrick Haley · Sarvesh Vinod Kumar · Kirtan Patel

Advisor: Prof. Amir Sadovnik

Repository: `github.com/joshcannonai/nba-game-intelligence-agent`

---

> **Draft, 2026-08-03.** Sections 1–8 are written. The models in §5.2 and the
> per-member breakdown in §6 are marked for their owners. Every number here came from
> running the code on 2026-08-03; Appendix A lists the command for each one.

---

### What we call things

Three different things in this report could be called "the model", so we don't call
any of them that.

| Term | What it means |
|---|---|
| **the predictor** | A small statistical model we trained. Takes two teams and a date, returns a percentage chance the home team wins. No AI writing involved. |
| **the agent** | An AI chatbot (like ChatGPT) that we gave a set of seven functions it can call to look things up. It reads what comes back and writes a report. |
| **a tool** | One of those seven functions. The agent can *only* use these — it can't browse the web or query a database. |
| **the closing line** | What professional sportsbooks think will happen, expressed as odds. This is our benchmark, because it's very hard to beat. |

---

## 1. Abstract

We built a system that predicts who will win NBA games, and then tested three
different ways of making that prediction to see which works best.

The first approach is a **predictor** — a small statistical model trained on two past
seasons. The second is an **agent** — an AI chatbot given seven lookup tools and asked
to reason its way to an answer. The third gives the agent the predictor's number and
lets it decide whether to agree.

We expected the third to win, since it has the most information. It came last. When
the agent disagreed with the predictor, it was wrong 15 times out of 19.

The whole thing rests on one rule: nothing the system sees may be dated after the game
it's predicting. That sounds obvious and is surprisingly easy to get wrong, especially
with an AI that may simply remember who won.

We also converted the results into money. Betting $100 a game on the predictor's picks
across a full season loses **$2,393**. That's a loss — but it's a smaller loss than
always betting the favourite, which loses $4,628 while winning *more* games.

## 2. Introduction

Predicting basketball games is not a new idea. Sportsbooks do it for money, and they
are good at it. We were never going to beat them, and that wasn't the goal.

The goal was to find out what we could actually build, and then measure honestly
which parts helped. Specifically: does adding an AI agent to a statistical prediction
make it better, worse, or no different? That question turns out to have a clear
answer, and it wasn't the one we expected.

**Why it's worth doing.** AI agents are being bolted onto existing systems all over
the place, usually assuming that helps. Sometimes it does. We had a setup where we
could actually check — same data, same games, one thing changing — and that method
transfers to anyone making the same call.

**The hard part.** Testing a normal prediction model is routine — you hide a season
from it during training, then see how it does. But an AI chatbot has read most of the
internet. If you ask it about a game from December 2025, it might just remember the
score.

That's a problem, because a system that remembers looks *fantastic* — near-perfect
scores, a demo that lands, and no actual prediction happening. So a lot of this project
is plumbing that stops the system reaching the answer. Without it, none of the numbers
mean anything.

## 3. Previous Work

**Public NBA forecasting.** FiveThirtyEight published NBA game predictions for years,
first using an Elo rating system (a chess-style rating where teams gain and lose points
based on results) and later a more detailed player-based model called RAPTOR. Their
methodology and past predictions are public. They had a full-time team; we had a
semester. We treat their work as a sanity check on what's realistic, not a target.

**Betting markets as a benchmark.** In sports economics the closing line — the final
odds before a game starts — is generally treated as a strong summary of everything
known about a matchup. That's why we use it as our benchmark rather than as an input.
This turned out to matter in practice, not just in theory (§5.4).

**AI agents that use tools.** Our agent follows a now-common pattern where the AI
alternates between thinking and calling a function, then reads the result before
deciding what to do next. This is the approach popularised as ReAct (Yao et al., 2023),
and we implemented it with the LangChain library. Nothing about the loop is novel. What
we changed is what the agent is *allowed* to reach.

**Testing AI on data it may have memorised.** This is a known and growing problem in AI
evaluation, which is why benchmark scores are increasingly published alongside the
model's knowledge cutoff date. Our version of the fix is simple: use an AI model whose
cutoff we can check, and only test on games that happened after it.

**How ours differs.** Existing NBA prediction work tries to be accurate and reports
accuracy. Agent research demonstrates what agents can do. We're doing neither — we hold
everything constant and vary only whether an AI is involved, then report what changed.
What changed was that it got worse, which isn't a result either field is set up to look
for.

> **Team note:** this section cites what we can actually stand behind. If anyone has
> read a specific paper on NBA prediction or betting market efficiency, add it — it's
> worth 10% and it's currently thin.

## 4. Dataset

Six sources, all committed to the repository so the project runs from a fresh download.

| Source | Where from | What we use it for | Size |
|---|---|---|---|
| NBA team stats, 1947–present | Kaggle | Last season's team ratings | 1,907 rows |
| NBA player stats, 1947–present | Kaggle | Player averages, injury weighting | 33,339 rows |
| NBA injuries, 2016–2025 | Kaggle | Who was hurt, up to Jan 2025 | 16,873 rows |
| NBA injuries, 2025 onward | ProSportsTransactions.com | Who was hurt during our test season | 3,581 rows |
| Betting odds, 2008–2026 | *source not recorded* | Our benchmark | 24,440 rows |
| Game schedules and results | NBA's own stats site | Who played who, and who won | ~1,300/season |

**Four problems we had to fix.**

*The injury data stopped too early.* The Kaggle set ends in January 2025 — before the
season we test on even starts. We extended it from ProSportsTransactions.com, which is
where that Kaggle set came from originally, so the columns line up. That site blocks
automated downloads, so the rows were collected by hand through a browser, 25 at a
time. The two files are kept separate so it's clear which rows came from where, and a
test checks they join with no gap and no double-counting.

*The odds file contains the answer.* Each row has the betting line and the final score
side by side. A lookup tool fetching the line would hand the agent the result. We built
a filtered copy that keeps a specific list of allowed columns, so scores can't get
through even if the original file changes.

*Season totals are misleading mid-season.* The Kaggle stats are end-of-season numbers.
Using them to predict a December game means using March data. We only use them for
"last completed season" comparisons; current form comes from game-by-game records.

*Two sources have no recorded origin.* The odds file and the file we used to verify it
were handed over without a link. Since our entire benchmark rests on that odds file,
that's a real gap and we're flagging it rather than quietly moving on. We did at least
confirm it contains *closing* odds rather than opening odds, by checking ten games
against an independent source: nine matched closing.

## 5. Technical Approach

### 5.1 Keeping the future out

Every lookup carries a date. Only information published on or before that date comes
back. If something can't be worked out, the answer is "unknown" with a reason — never
zero. That distinction matters: an empty injury list because nobody is hurt and an
empty injury list because we don't know are very different, and treating the second as
the first produces a confident wrong answer.

We block the future twice.

First, a plain script copies the data into a fresh folder with everything after the
chosen date stripped out. No AI involved. Point the agent at that folder and the future
isn't filtered — it isn't there.

Second, each tool applies its own date check on top. This isn't belt-and-braces. The
two tools genuinely need different cutoffs: recent-form needs games strictly *before*
the date, while schedule lookup needs games *through* it. One folder can't satisfy both
without breaking one of them.

One thing that is *not* cheating: knowing when games are scheduled. The NBA publishes
its calendar in August, so "this team played last night" is fair game on any date. Only
the *results* are hidden.

Third — and separately — we had to deal with what the AI already knows. Blocking data
doesn't help if the chatbot remembers the score. So scored runs use a model that runs
on a laptop with a knowledge cutoff of around January 2025, which is before every game
we test on.

### 5.2 The predictor

The predictor is a **logistic regression**. In plain terms: it takes eight numbers
about a matchup, multiplies each by a weight, adds them up, and squashes the total into
a percentage between 0 and 100.

We picked it over fancier options on purpose. You can read its weights and argue with
them, which you cannot really do with a large ensemble of decision trees. It saves to a
few hundred bytes of plain text instead of an opaque binary file. And it runs anywhere
without heavyweight machine-learning libraries installed.

The eight inputs are all *differences* between the two teams: win percentage, average
scoring margin over the last ten games, days of rest, whether either played yesterday,
injury load, and games played so far. Every one is calculated only from games that
happened *before* the game being predicted.

That last point is where this kind of project usually breaks. The obvious way to
calculate a team's win percentage is to take their season record — but the season
record includes the game you're trying to predict. You end up with a model that looks
astonishingly accurate and is worthless. Our code updates every running total *after*
recording each game, never before.

We also split the data by **season**, not randomly. A random split lets the model learn
from March games to predict January ones, which inflates the score by a few points that
disappear the moment anybody checks properly.

> **Sarvesh:** your two models go here. For each one, say plainly what goes in and what
> comes out, how you split train and test, and what you compared against. The 07-28
> review flagged that the win classifier might be set up as a regressor — sorting that
> out is the most useful thing in this section.

### 5.3 How we score it

Three measures, because "percentage of games called right" hides a lot.

- **Accuracy** — how often the pick was right.
- **Log loss** — punishes being confident *and* wrong. Saying "95% sure" and being wrong hurts far more than saying "51% sure" and being wrong.
- **Brier score** — how far the predicted percentage was from what actually happened, on average.

Two reference points: always picking the home team (they win about 55% of the time),
and the closing line.

### 5.4 The three approaches

- **A — the predictor alone.** No AI.
- **B — the agent alone.** It gets the lookup tools but no predictor, and has to reason its way to a number.
- **C — the agent plus the predictor.** Same as B, but it also gets the predictor's answer.

B and C are the same AI, same instructions, same data. The only difference is whether
the predictor's number is available. A test enforces that.

**One thing we had to remove.** The agent originally had a tool for looking up the
betting line. Its instructions said, in capital letters, that the line was background
only. Running it live, we watched it write *"The closing betting line favors the home
team, ORL (-5.5 spread)"* into its own reasoning.

The betting line is what we grade it against. An agent that reads the line and repeats
it scores brilliantly and has predicted nothing. Nothing dramatic happened here — it
was asked what informed its answer and honestly said. The tool just made it possible.
Telling a model not to use something is a request. Taking the tool away is a guarantee.

## 6. Implementation

Python 3.11+. The agent uses **LangChain**; the predictor is trained with
**scikit-learn** and saved as plain JSON; data work uses **pandas**; the interface is
**Streamlit**; tests use **pytest**. Scored agent runs use **Gemma 4** through
**Ollama**, running locally with no API key and no cost.

The repository splits by job: `agent/` (the tools and the loop), `models/` (the
predictor), `eval/` (the scoring harnesses), `scripts/` (data preparation and the date
filter), `skills/`, `ui/`, and `tests/`.

**The seven tools** are the agent's whole world: matchup context, team form, injuries,
player splits, the predictor's number, the schedule, and projected player stats. Two of
those seven return "not built yet" — we never finished them — and the agent reports
that rather than making something up. Running `python -m agent.run --status` prints
what works and what doesn't, generated from the code rather than a list we maintain.

**Skills.** Each tool has a plain-English rules file saying when to use it and how to
read the result. These get loaded into the agent's instructions at startup, which means
a teammate can change how the agent behaves by editing a text file.

**Why not a database.** We considered letting the agent query a database freely, and
didn't — for the same reason we removed the betting-line tool. An open query surface is
how a system reaches data nobody meant it to have.

**Testing.** 73 tests. Rather than just check that a filter runs, we deliberately broke
each rule and confirmed the tests noticed — feeding future data in early (3 tests
caught it), letting two parts of the code drift apart (1), and training on the test
season (3). Another test launches a separate process to confirm the agent really reads
the filtered folder.

## 7. Results

### 7.1 Accuracy

The predictor, on all 1,322 games of the 2025-26 season, which it never trained on:

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.687 | 0.247 |
| **the predictor** | **66.5%** | **0.612** | **0.212** |
| closing line | 69.0% | 0.578 | 0.198 |

It scored 66.8% on the data it trained on and 66.5% on data it had never seen — a gap
of 0.3%, which means it learned patterns rather than memorising games. It sits 2.5
points behind the sportsbooks.

Worth being honest about: a hand-tuned rule of thumb we wrote earlier already scored
66.3%. On raw accuracy the trained predictor barely improves on it. What it does better
is express *confidence* sensibly, and it can be checked by tests rather than tuned by
feel.

### 7.2 All three approaches

The AI approaches take about 40 seconds per game, so they ran on two 40-game samples
rather than the full season, with every approach scored on the same games.

| | A — predictor | B — agent alone | C — agent + predictor |
|---|---|---|---|
| sample 1 | 75.0% | 57.5% | 55.0% |
| sample 2 | 70.0% | 62.5% | 62.5% |

**C came last.** Since 40-game accuracy is noisy, the more reliable question is: on the
games where the agent *disagreed* with the predictor it was handed, who was right?
Pooling both samples, it disagreed 19 times and was wrong on 15 of them.

We then gave each tool written rules — including an explicit instruction not to
second-guess the predictor on injuries — and re-ran the same games:

| pooled, 80 games | before | after |
|---|---|---|
| C's accuracy | 58.8% | **66.3%** |
| disagreements | 19 | **11** |

Roughly half the gap closed, and it held on both samples. This is encouraging rather
than settled: with only 11 disagreements left there isn't enough data to call it
statistically solid.

### 7.3 What it's worth in money

Accuracy is abstract. So we asked a concrete question: **bet $100 on whoever each
approach picked. Who ends up ahead?**

One caveat first, and we checked it rather than waving at it. Our odds file has no
moneyline prices for this season, only spreads, so prices had to be reconstructed from
the spread. The scores and spreads are real; the prices are calculated.

To find out whether that calculation is trustworthy, we tested it against the **19,807
earlier games in the same file that do carry real quoted prices**. Reconstructing those
from their spreads and comparing to what bookmakers actually offered:

| | |
|---|---|
| correlation with real prices | **0.9959** |
| average error | 2.9 percentage points |
| within 5 points | 77.6% of games |

The house margin is measured from those same games (3.75%) rather than assumed. So the
ranking below is solid — every approach faces identical prices on identical games — and
the dollar figures carry roughly a 3-point pricing error.

**Full season, 1,322 games:**

| approach | games won | profit | return |
|---|---|---|---|
| **the predictor** | 66.4% | **−$2,393** | −1.8% |
| always bet the favourite | 69.0% | −$4,628 | −3.5% |
| always bet the home team | 55.5% | −$7,350 | −5.6% |

Everything loses money. That is the correct result and the most useful number in this
report — the house margin is the bar, and none of our approaches clears it.

But look at the top two rows. **Always betting the favourite wins more games (69.0% vs
66.4%) and loses nearly twice as much money.** Favourites win often and pay badly. Our predictor picks
more selectively, so it loses less. That gap is the entire argument for why accuracy
alone is a poor way to judge a prediction system.

**On the 80 games where we could run all three approaches:**

| approach | profit | return |
|---|---|---|
| A — predictor | +$993 | +12.4% |
| C — agent + predictor (after rules) | +$77 | +1.0% |
| C — agent + predictor (before rules) | −$832 | −10.4% |
| B — agent alone | −$861 | −10.8% |

**Do not read A's +$993 as a profitable system.** The same predictor loses $2,393 over
the full season. Those 80 games happened to suit it — which is exactly the trap this
whole report is about, and we'd rather demonstrate it on ourselves than pretend we
found an edge. The genuinely interesting number is C moving from −$832 to roughly
break-even once we constrained it.

### 7.4 A measurement that changed our design

We intended to write a rule like "star player out, drop the prediction by N%". There
is no N that the data supports, and finding that out was instructive.

| comparison | result |
|---|---|
| all games | teams win *more* without their star — clearly wrong |
| only teams that have a star player | +5.6% more wins without him |
| **each team compared against itself** | **+0.0%** |

The first two are both distorted the same way. Having a 20-point-per-game scorer is a
property of *good teams*, and a good team is still good on the night that player sits
out. Comparing a team only against itself removes that, and the effect vanishes.

## 8. Conclusion and Future Work

We built a working NBA prediction system, tested three ways of making the prediction,
and measured which was best. The statistical predictor won. Adding an AI agent on top
made it worse, mostly by talking itself out of correct answers, and giving that agent
explicit written rules recovered about half the damage.

None of the approaches makes money against realistic betting prices. The predictor
loses least, and notably loses less than a strategy that wins more games.

**What we'd tell someone repeating this.** Verify your AI hasn't seen the test data
rather than assuming. Change one thing at a time. Convert accuracy into whatever unit
actually matters — money, here — because a system can be more accurate and still worse.
And distrust small samples, including flattering ones.

**Limitations.** The AI comparisons rest on 80 games, which is not many. Our injury
data records when players were placed on the injured list, not when the news broke, so
a same-day entry can slip in. We link players to teams using last season's rosters, so
"injured" and "changed teams" aren't cleanly separated. Two data sources have no
recorded origin. And projected player statistics, which we promised in the proposal,
were never built.

**Future work.** The most immediate step is reading the agent's own reasoning on the 19
disagreements to test our theory that it over-weights injuries. Beyond that: adjusting
for strength of schedule, which is the biggest single improvement available; using
official timestamped injury reports instead of transaction records; running the AI
comparisons on far more games; and the experiment our advisor suggested — deliberately
letting the agent see the future, to measure exactly what cheating is worth.

## 9. References

1. Datta, S. *NBA/ABA/BAA Stats (1947–present)*. Kaggle.
2. Oberweis, J. *2016–2025 NBA Injury Data*. Kaggle.
3. ProSportsTransactions.com — basketball transactions archive. Retrieved 2026-07-28.
4. NBA Stats API (`stats.nba.com`), via the `nba_api` Python package.
5. Basketball Reference, via `basketball_reference_web_scraper`.
6. ESPN NBA injuries API.
7. Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
8. FiveThirtyEight NBA forecasts — Elo and RAPTOR methodology and published predictions.
9. LangChain — agent framework. Ollama / Gemma 4 — local AI model.
10. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12 (2011).

> **Team note:** 7, 8 and 10 are real and correctly attributed, but nobody has read 7
> or 10 properly. Either read them or drop them — a citation you can't discuss is worse
> than a shorter list.

---

## Appendix A — Reproducing the numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                 # 73 tests
python -m models.train                 # §5.2 weights, §7.1 accuracy
python -m eval.three_arms              # §7.1 predictor vs baselines
python -m eval.betting                 # §7.3 the money
python -m eval.injury_impact           # §7.4 the star-player measurement
python eval/crosscheck_odds.py         # §4 closing-vs-opening check
python -m scripts.gate_snapshot --as-of 2026-01-14   # §5.1 the date filter
python -m agent.run --status --source real           # §6 what works

# the agent itself, running locally (needs `ollama serve`, `ollama pull gemma4`)
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```
