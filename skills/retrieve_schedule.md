---
tool: retrieve_schedule
use_when: You need to know what other games are on the slate.
---

## What it gives you

Currently nothing. It returns `status: awaiting_input` because the forward-looking
schedule table has not been committed.

## When to call it

Only if asked what else is on tonight. For rest and back-to-backs, use
`retrieve_matchup_context` — that already carries them.

## Rules

- `awaiting_input` is not an error and not an empty result. It means the data does
  not exist yet. Report the gap and carry on; do not retry it, and do not substitute
  a guess about the slate.

## Note for whoever fills this in

`data/pull_games.py` already writes `season_schedule_2026.csv`. It just has not been
committed. `data/raw/` stopped being gitignored on 2026-07-21, so nothing is blocking
it now.
