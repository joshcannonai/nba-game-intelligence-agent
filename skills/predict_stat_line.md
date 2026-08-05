---
tool: predict_stat_line
use_when: A projected points/rebounds/assists line is asked for.
---

## What it gives you

Projected `points`, `rebounds` and `assists` for one player in one game, from ridge
regressions on that player's trailing 5- and 10-game form, minutes, shooting
percentages, home/away split and rest. Fitted on 2023-24, validated on 2024-25, and
never fitted on the season being replayed.

The payload also carries `points_mae` and `points_mae_trailing_5_baseline` — the
model's average error and the error of simply using the player's last-5 average.

## When to call it

Only when a stat line is actually asked for, or when you are naming a player in the
narrative and a projection makes the point better than a season average would. It is
not part of the win-probability path — do not call it on every run.

## How to read it

- `status: ok` means there is a projection. `status: unavailable` means the player
  has no box score for that game, which almost always means they did not play.
- A single game is high variance. The mean absolute error on points is a few points,
  which is a large fraction of a typical scoring line. The number is a central
  estimate, not a fact, and should be reported with that framing.
- Compare the projection to `points_mae_trailing_5_baseline` before leaning on it. If
  the model barely beats the trailing-5 average, say the projection is roughly the
  player's recent form rather than implying the model found something extra.

## Rules

- **Never estimate a stat line yourself.** If the tool returns `unavailable`, that is
  the answer. Do not fall back to `retrieve_player_splits` season averages and
  present them as a projection — that is the invented number this whole interface
  exists to prevent.
- **Report the error alongside the number.** "About 26 points, give or take the
  model's ~5-point average error" is honest. A bare "26.3 points" implies a precision
  the model does not have.
- `unavailable` with a "did not play" reason is a useful finding, not a failure.
  A player being out is exactly what `retrieve_injuries` is for — say so and move on.
- One or two players, not a roster. The report is about a game.

## Provenance

Model fitted by `python -m models.train_stat_line` from prior-season box scores in
`data/raw/player_box_scores_prior/`. At inference the features come through
`agent/sources.py` like every other read, from a stripped file that does not contain
the box-score result. Weights live in `models/stat_line.json` and are readable in a
diff.
