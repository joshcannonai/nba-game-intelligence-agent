# Agent skills — review copy

**NBA Game Intelligence Agent · CECS 499 · generated 2026-08-03**

Each of the agent's tools has a *skill*: a short set of rules telling it when to call
that tool and what to do with the answer. The agent loads these at startup, so these
rules are what it actually follows — this is not documentation written after the fact.

**Patrick — this is the doc from Tuesday's call.** Read it as "is this what we want the
agent to do?" You do not need to touch any code to have an opinion here. Comment
directly on anything that looks wrong, missing, or too strict, and I will move the
changes back into the repo.

Two places I would especially like a second opinion:

1. **`retrieve_injuries`** — we agreed on the call to encode something like "if a
   player averages over 20 ppg and is out, drop the odds by X%". I could not find an
   X the data supports, so the rule currently says the opposite: report the injury and
   let the fitted model price it. The measurement behind that is in the skill itself.
   If you think that is the wrong call, say so.
2. **`retrieve_team_form`** — the "under 5 games is noise" threshold is my judgement,
   not a measured number.

Source of truth is `skills/` in the repo. Regenerate this document with
`python scripts/skills_doc.py`.

---

## The tools at a glance

| Tool | Use when |
|---|---|
| `retrieve_matchup_context` | Always, first, before any other tool. |
| `retrieve_player_splits` | A specific player's production is load-bearing in your explanation. |
| `retrieve_schedule` | You need to know what other games are on the slate. |
| `retrieve_team_form` | You need current strength rather than last season's. |
| `retrieve_injuries` | You need who was unavailable on the morning of the game. |
| `predict_stat_line` | A projected points/rebounds/assists line is asked for. |
| `predict_win_probability` | Always. This is the number the report is built around. |

---


## `retrieve_matchup_context`

**Use when:** Always, first, before any other tool.

## What it gives you

Both teams' ratings, days of rest, back-to-back flags, the known injury list, and
head-to-head results — all filtered to `as_of_date`.

### When to call it

First, every time. It is the cheapest way to see the whole picture, and the other
tools are follow-ups on what it shows.

### How to read it

- `basis` tells you where team strength came from. `"prior completed season"` means
  the current season had too few games and it fell back — treat that strength number
  as stale, especially after December.
- `rest` is computed from the published schedule, not from results. It is trustworthy
  on any date.
- `warnings` is not decoration. If it says something is unavailable, that thing is
  unknown, not zero.

### Rules

- Never treat a `null` as a zero. An empty injury list with a warning means "we do
  not know", which is different from "nobody is hurt".
- If `h2h_last_5` is empty, the teams have not met yet this season. Say so; do not
  reach for last season's meetings as though they were this season's.


## `retrieve_player_splits`

**Use when:** A specific player's production is load-bearing in your explanation.

## What it gives you

Season averages for one player, and optionally their back-to-back split.

### When to call it

Sparingly. Only when you are naming a player in the narrative and need their actual
numbers, or when a team is on a back-to-back and you want the fatigue split.

### How to read it

The split is only meaningful with enough back-to-back games behind it. If the tool
says the split is unavailable, it is unavailable — that is not the same as "no
fatigue effect".

### Rules

- One or two players, not a roster. The report is about a game, not a team profile.
- Averages are from the prior completed season unless stated. Do not present them as
  this season's form.


## `retrieve_schedule`

**Use when:** You need to know what other games are on the slate.

## What it gives you

Currently nothing. It returns `status: awaiting_input` because the forward-looking
schedule table has not been committed.

### When to call it

Only if asked what else is on tonight. For rest and back-to-backs, use
`retrieve_matchup_context` — that already carries them.

### Rules

- `awaiting_input` is not an error and not an empty result. It means the data does
  not exist yet. Report the gap and carry on; do not retry it, and do not substitute
  a guess about the slate.

### Note for whoever fills this in

`data/pull_games.py` already writes `season_schedule_2026.csv`. It just has not been
committed. `data/raw/` stopped being gitignored on 2026-07-21, so nothing is blocking
it now.


## `retrieve_team_form`

**Use when:** You need current strength rather than last season's.

## What it gives you

A rolling 10-game record and average point differential, using only games played
before `as_of_date`.

### When to call it

When `retrieve_matchup_context` reports `basis: prior completed season`, or whenever
you are about to make a claim about how good a team is *right now*.

### How to read it

`avg_point_diff` is the useful number. Roughly: +5 is a strong team, 0 is average,
−5 is weak. It is a better guide in December than a prior-season rating, which is
describing a roster that may no longer exist.

### Rules

- Under 5 games played, the window is noise. Say the sample is small rather than
  reporting the number as though it were settled.
- This is *not* opponent-adjusted. A 5–0 run against weak teams looks identical to
  5–0 against strong ones. If a team's record and its point differential disagree,
  trust the differential and say why.


## `retrieve_injuries`

**Use when:** You need who was unavailable on the morning of the game.

## What it gives you

Players known to be out as of that morning, sorted by `importance` (a prior-season
minutes-and-points proxy), each with `days_out`.

### When to call it

Once per team, when the matchup context flags injuries or when you are about to
explain a prediction that hinges on availability.

### How to read it

- `importance` is `None`, not `0.0`, for a player with no prior season. A rookie is
  **unknown**, not worthless. Do not sum it without handling `None`.
- `days_out` above ~60 usually means the team has already adjusted. A player out
  since November is priced into their recent form; counting it again double-counts.
- These are **transaction dates**, not news timestamps — when a player was placed on
  or activated from the injured list. A same-day placement can appear here.

### Rules

- **Do not apply your own injury penalty on top of the model's number.** This is the
  most important rule in this directory, and it is measured, not a preference.

  We tested the obvious rule — "a player averaging over 20 ppg is out, so drop the
  odds by N%" — and could not find an N that the data supports. Comparing each team
  against **itself**, with its top scorer versus without, the difference in win rate
  is **+0.0% (standard error 3.3%, n = 21 teams)**. The spread across teams runs from
  −32% to +36%. A pooled comparison across teams looks significant (+5.6%, z = 2.6)
  and points the wrong way, because having a 20 ppg scorer is a property of good
  teams. Reproduce with `python -m eval.injury_impact`.

  The fitted model already carries an injury term (`injury_weight_diff`, standardised
  weight −0.246) learned over two seasons. Your job is to *report* the injury list,
  not to re-price it.

- This matters because we measured the cost of ignoring it. When the agent overruled
  the model, it was wrong **15 times out of 19** (`docs/REPORT.md` §9.6), and
  over-weighting the injury list is our leading explanation.
- Report who is out and how much they played. Stop there.


## `predict_stat_line`

**Use when:** A projected points/rebounds/assists line is asked for.

## What it gives you

Currently nothing — `status: awaiting_input`. The regression behind it was never
built.

### Rules

- Report it as missing. Never estimate a stat line yourself from season averages and
  present it as a projection; that is exactly the invented number this whole
  interface exists to prevent.
- Projected stat lines were a stated deliverable in the proposal. Saying "not built"
  is correct and expected. Saying "LeBron will score about 25" is not.


## `predict_win_probability`

**Use when:** Always. This is the number the report is built around.

## What it gives you

`home_win_prob` from a logistic regression trained on 2023-24 and 2024-25, plus the
eight features it used and its holdout accuracy.

### When to call it

Every run, after `retrieve_matchup_context`.

### How to read it

It scored **66.5%** on all 1,322 games of 2025-26, a season it never trained on,
against 55.5% for always picking the home team and 69.0% for the Vegas closing line.
It sees form, win percentage, rest, back-to-backs and injury load — nothing your
retrieval tools cannot also show you.

### Rules

- **Treat its number as the answer unless you have a concrete, specific reason it is
  wrong** — and if you move off it, say exactly why in `key_factors`.
- This is not deference for its own sake. We measured it. Across two paired 40-game
  samples the agent overruled the model 19 times and was wrong on 15 of them
  (two-sided sign test p ≈ 0.019). Overruling made the prediction worse, reliably.
- Good reasons to move off it: a tool returned something the model provably cannot
  see. Bad reasons: the injury list looks alarming (see
  `skills/retrieve_injuries.md`), the favourite "feels" wrong, or the probability
  seems too confident.
- If it returns `awaiting_input`, the model file is missing. Say so. Do not
  substitute your own estimate.
