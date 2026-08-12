---
tool: retrieve_matchup_context
use_when: Always, first, before any other tool.
---

## What it gives you

Both teams' ratings, days of rest, back-to-back flags, the known injury list, and
head-to-head results — all filtered to `as_of_date`.

## When to call it

First, every time. It is the cheapest way to see the whole picture, and the other
tools are follow-ups on what it shows.

## How to read it

- `basis` tells you where team strength came from. `"prior completed season"` means
  the current season had too few games and it fell back — treat that strength number
  as stale, especially after December.
- `rest` is computed from the published schedule, not from results. It is trustworthy
  on any date.
- `warnings` is not decoration. If it says something is unavailable, that thing is
  unknown, not zero.

## Rules

- Never treat a `null` as a zero. An empty injury list with a warning means "we do
  not know", which is different from "nobody is hurt".
- If `h2h_last_5` is empty, the teams have not met yet this season. Say so; do not
  reach for last season's meetings as though they were this season's.
- Head-to-head is descriptive context, not a primary prediction signal. A few games
  between changing rosters should not outweigh current rolling form.
- Rest and back-to-backs are tie-breakers. Do not let a one-day rest edge overwhelm
  a clear difference in rolling point differential and current record.
