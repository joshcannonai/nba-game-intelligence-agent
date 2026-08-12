# Agent skills — review copy

**NBA Game Intelligence Agent · CECS 499 · generated 2026-08-11**

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
| `predict_win_probability` | After matchup context. Adds Model A output as one additional data point for Model C. |

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
- Head-to-head is descriptive context, not a primary prediction signal. A few games
  between changing rosters should not outweigh current rolling form.
- Rest and back-to-backs are tie-breakers. Do not let a one-day rest edge overwhelm
  a clear difference in rolling point differential and current record.


## `retrieve_player_splits`

**Use when:** A specific player's production is load-bearing in your explanation.

## What it gives you

Season averages for one player from a season completed before `as_of_date`, and
optionally their back-to-back split.

### When to call it

Only for an explicit player question. Do not call it for the team-level winner
prediction. Always pass the authorized run cutoff as `as_of_date`.

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

The fixtures tipping off in the days after `as_of_date`: matchup id, date, away and
home. Read from the season's game log, filtered to the window.

### When to call it

Only if asked what else is on tonight, or when the slate itself is part of the
answer. For rest and back-to-backs use `retrieve_matchup_context`, which already
carries them. This is not part of the win-probability path.

### How to read it

- `count` is how many games are in the window, not how many exist in the season.
- Teams and dates only. There are no tip-off times in the dataset, so if you are
  asked when a game starts, say that is not available rather than guessing.

### Rules

- **Never report a score or a winner from this tool.** It does not return them, and
  that is deliberate: the rows it reads carry `home_pts`, `away_pts` and `winner`
  right next to the fixture, and only the identity fields are copied out.
- Knowing *who plays whom* on a future date is not leakage. The NBA publishes its
  schedule in August, so a fixture is knowable on any as-of date. Knowing *how it
  went* is leakage. Keep that line.


## `retrieve_team_form`

**Use when:** You need current strength rather than last season's.

## What it gives you

A rolling 10-game record and average point differential, using only games played
on or before `as_of_date`.

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
- For Model B, compare both teams and treat the difference in rolling point
  differential as the primary strength signal. Use current win percentage only as
  corroboration. Start from the 55% home-team base rate, keep an otherwise even
  matchup near that base rate, and reserve 60-70% for cases where form margin and
  record agree strongly.


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

- **Do not invent a numeric injury penalty.** Current form already reflects established
  absences, and Model C's predictor has its own learned injury-load feature. Use the
  list to explain availability and uncertainty, not to apply an unsupported formula.
- Report who is out and how much they played. Stop there.


## `predict_stat_line`

**Use when:** A projected points/rebounds/assists line is asked for.

## What it gives you

Projected `points`, `rebounds` and `assists` for one player in one game, from ridge
regressions on that player's trailing 5- and 10-game form, minutes, shooting
percentages, home/away split and rest. Fitted on 2023-24, validated on 2024-25, and
never fitted on the season being replayed.

The payload also carries `points_mae` and `points_mae_trailing_5_baseline` — the
model's average error and the error of simply using the player's last-5 average.

### When to call it

Only when a stat line is actually asked for, or when you are naming a player in the
narrative and a projection makes the point better than a season average would. It is
not part of the win-probability path — do not call it on every run.

### How to read it

- `status: ok` means there is a projection. `status: unavailable` means either the
  gated injury report lists the player as out or the gated history lacks enough
  observable pregame inputs. It does not reveal any later participation.
- A single game is high variance. The mean absolute error on points is a few points,
  which is a large fraction of a typical scoring line. The number is a central
  estimate, not a fact, and should be reported with that framing.
- Compare the projection to `points_mae_trailing_5_baseline` before leaning on it. If
  the model barely beats the trailing-5 average, say the projection is roughly the
  player's recent form rather than implying the model found something extra.

### Rules

- **Never estimate a stat line yourself.** If the tool returns `unavailable`, that is
  the answer. Do not fall back to `retrieve_player_splits` season averages and
  present them as a projection — that is the invented number this whole interface
  exists to prevent.
- **Do not project a player listed out.** Check `retrieve_injuries` first when
  availability matters. The tool also enforces this boundary and returns
  `unavailable` if the gated injury report already lists the requested player out.
- **Report the error alongside the number.** "About 26 points, give or take the
  model's ~5-point average error" is honest. A bare "26.3 points" implies a precision
  the model does not have.
- Never infer future participation from `unavailable`. Use `retrieve_injuries` for
  availability information known by the as-of date.
- One or two players, not a roster. The report is about a game.

### Provenance

Model fitted by `python -m models.train_stat_line` from prior-season box scores in
`data/raw/player_box_scores_prior/`. At inference the features come through
`agent/sources.py` like every other read. It recomputes them from outcome-history
rows dated on or before `as_of`; structural snapshots physically remove later rows.
Weights live in `models/stat_line.json` and are readable in a diff.


## `predict_win_probability`

**Use when:** After matchup context. Adds Model A output as one additional data point for Model C.

## What it gives you

The exact Model A `home_win_prob` shown by the UI, plus the pregame feature values,
training seasons, and holdout accuracy recorded with the fitted model.

### When to call it

Every run, after `retrieve_matchup_context`.

### How to read it

It is a logistic regression over rolling point margin, win percentage, rest,
back-to-backs, injury load, and games played. The fitted weights use only the two
training seasons. Live feature inputs stop at the requested cutoff. This tool and
the UI's Model A button call the same function.

### Rules

- Treat `home_win_prob` as peer evidence alongside the matchup, team-form, injury,
  and rest outputs. It is one additional data point, not the starting answer.
- Synthesize the final probability from the complete gated evidence. Model C may
  agree or disagree with Model A.
- State a material agreement or disagreement and the evidence behind it in
  `key_factors`.
- If it returns an error or unavailable status, report that in `missing`. Do not
  substitute your own version of Model A.
