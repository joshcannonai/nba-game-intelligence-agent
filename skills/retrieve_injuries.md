---
tool: retrieve_injuries
use_when: You need who was unavailable on the morning of the game.
---

## What it gives you

Players known to be out as of that morning, sorted by `importance` (a prior-season
minutes-and-points proxy), each with `days_out`.

## When to call it

Once per team, when the matchup context flags injuries or when you are about to
explain a prediction that hinges on availability.

## How to read it

- `importance` is `None`, not `0.0`, for a player with no prior season. A rookie is
  **unknown**, not worthless. Do not sum it without handling `None`.
- `days_out` above ~60 usually means the team has already adjusted. A player out
  since November is priced into their recent form; counting it again double-counts.
- These are **transaction dates**, not news timestamps — when a player was placed on
  or activated from the injured list. A same-day placement can appear here.

## Rules

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
