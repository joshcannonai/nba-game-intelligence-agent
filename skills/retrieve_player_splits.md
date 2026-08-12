---
tool: retrieve_player_splits
use_when: A specific player's production is load-bearing in your explanation.
---

## What it gives you

Season averages for one player from a season completed before `as_of_date`, and
optionally their back-to-back split.

## When to call it

Only for an explicit player question. Do not call it for the team-level winner
prediction. Always pass the authorized run cutoff as `as_of_date`.

## How to read it

The split is only meaningful with enough back-to-back games behind it. If the tool
says the split is unavailable, it is unavailable — that is not the same as "no
fatigue effect".

## Rules

- One or two players, not a roster. The report is about a game, not a team profile.
- Averages are from the prior completed season unless stated. Do not present them as
  this season's form.
