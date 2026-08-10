# Proposal: weighting ideas for `predict_win_probability` (for Sarvesh's `models/` lane)

> Historical proposal. The current fitted model and its measured weights live in
> `models/predict.py` and `models/win_probability.json`.

**Status: archived design history, not current implementation guidance. The
placeholder discussed below was removed; current ownership and behavior are
documented in `models/README.md` and `docs/tool-contracts.md`.**

## What the historical placeholder (`stub_net_rating_v0`) did

`net rating differential + home-court constant (+2.5) + rest edge`,
squashed into a probability and clamped to [0.15, 0.85]. Its own docstring
calls out the biggest gap: **it ignores the injury list entirely**, so the
prediction doesn't move when the injury list does -- a team missing its
best player and a team at full health get the same number if their season
ratings are equal.

## A bug identified in that retired placeholder

The removed `_stub_win_probability` real-data branch hardcoded
`rest_edge = 0.0` with the comment "real rest needs game logs; do not
guess." That comment predates `scripts/fetch_game_logs.py` landing
`data/samples/game_logs_2024.csv` / `game_logs_2025.csv` -- real rest data
now exists and `agent/sources.py`'s `schedule_context()` already computes
`home_days_rest` / `away_days_rest` / back-to-back flags from it. The stub
did not call it for the real-data path.

This is preserved as the design observation that motivated explicit schedule
features. It is not an instruction to restore or edit the retired helper.

## Signals that were available but absent from the stub

All already retrievable through existing, tested tools -- this is a
"what's sitting there unused" list, not a request for new data collection:

1. **Injuries** (`retrieve_injuries` / `injuries_as_of`) -- currently
   unused by the probability itself. Simplest version: a fixed penalty per
   player out (crude, treats a 10th man and a starter the same -- the code
   already documents this exact limitation). Better version: scale the
   penalty by the player's `pts_avg` / team `off_rating` share from
   `player_season_averages`, so a leading scorer being out moves the
   number more than a bench piece. Still not "true" injury impact (that
   needs on/off-court splits this data doesn't have), but strictly better
   than ignoring it.
2. **Real rest/back-to-back** (see the bug above) -- already computed,
   just not wired to the real-source branch.
3. **Head-to-head history** (`h2h_last_5` via `schedule_context`) -- small
   sample (5 games), so worth a light touch (e.g. a capped +/- adjustment)
   rather than a strong weight; 5-game H2H records are noisy and can
   overfit if trusted too much.
4. **Recency / current-season form** -- the honest gap: `team_ratings`
   only serves the *prior completed* season (leakage-safe by design), so
   there's no in-season trend signal today. `retrieve_team_form` (the
   rolling as-of rating, TODO'd to Patrick+Kirtan) is the real fix; until
   it exists, there isn't a leakage-safe way to add this.

## Bigger picture: heuristic vs. trained model

All four items above are still hand-tuned weights, the same category of
thing the current stub already is -- useful for closing obvious gaps
quickly, but not what "properly weighted" should mean by the time this is
presented as the finished model. The actual ask (a model that's *learned*
the right weights from historical outcomes, not guessed at them) is what
XGBoost against `nba_stats_1947_present` + the game logs is for. The
suggestion here is sequencing, not a substitute:

1. Quick wins now: fix the rest bug, add a crude injury penalty. Cheap,
   removes the two most obviously-wrong blind spots, keeps the placeholder
   honest while the real model is being built.
2. The XGBoost model replaces the whole heuristic, trained against
   `nba_stats_1947_present` win/loss outcomes with these same signals
   (net rating, rest, injuries, H2H) as engineered features -- at which
   point the "properly weighted" question has an actual, defensible
   answer: what the trained model learned, which you can show and explain
   (feature importances), not what a human guessed.
3. Once that model exists, evaluate it through the maintained `eval/betting.py`
   path. `proposals/sportsbook-backtest-prototype.md` preserves the historical
   heuristic results without presenting the retired prototype as runnable.

## Not proposing

A finished weighting scheme or trained model -- that's real modeling work
requiring decisions about features, validation splits, and overfitting
checks that are Sarvesh's to make and defend, not something to hand him
pre-built.
