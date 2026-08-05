---
tool: retrieve_schedule
use_when: You need to know what other games are on the slate.
---

## What it gives you

The fixtures tipping off in the days after `as_of_date`: matchup id, date, away and
home. Read from the season's game log, filtered to the window.

## When to call it

Only if asked what else is on tonight, or when the slate itself is part of the
answer. For rest and back-to-backs use `retrieve_matchup_context`, which already
carries them. This is not part of the win-probability path.

## How to read it

- `count` is how many games are in the window, not how many exist in the season.
- Teams and dates only. There are no tip-off times in the dataset, so if you are
  asked when a game starts, say that is not available rather than guessing.

## Rules

- **Never report a score or a winner from this tool.** It does not return them, and
  that is deliberate: the rows it reads carry `home_pts`, `away_pts` and `winner`
  right next to the fixture, and only the identity fields are copied out.
- Knowing *who plays whom* on a future date is not leakage. The NBA publishes its
  schedule in August, so a fixture is knowable on any as-of date. Knowing *how it
  went* is leakage. Keep that line.
