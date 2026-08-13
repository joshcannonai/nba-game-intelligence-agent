---
tool: predict_best_player
use_when: You need the highest projected-points player in this matchup.
---

## What it gives you

The player on either side with the highest `predict_stat_line` points projection,
plus a short ranked list. Candidates are the gated rotation: players last seen on
these two teams on or before `as_of_date`, with injured players skipped.

## When to call it

When someone asks who to watch, who scores, or who the best player is tonight.
It is not part of the required winner-path retrievals. Do not call it to decide
who wins the game.

## How to read it

- `status: ok` means there is a ranked projection. `status: unavailable` means
  no gated rotation player produced a line.
- `points_mae` is the model's average error. Report it with the number.
- `uses: predict_stat_line` is the whole implementation. There is no second model.

## Rules

- **Never pick a best player yourself** from season averages or memory.
- **Do not use this to move the win probability.** It answers a player question.
- Report the projection as a central estimate, not a fact.
