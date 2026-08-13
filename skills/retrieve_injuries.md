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

- **Do not invent a numeric injury penalty.** Current form already reflects established
  absences, and Model C's predictor has its own learned injury-load feature. Use the
  list to explain availability and uncertainty, not to apply an unsupported formula.
- Rank absences by `importance` when it is not None. A high-importance player is a
  larger availability concern than a low-importance one (Kyrie vs a backup center).
  Do not treat every name on the list as equal.
- Report who is out and how much they played. Stop there.
