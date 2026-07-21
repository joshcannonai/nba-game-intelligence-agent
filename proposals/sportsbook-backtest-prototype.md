# Proposal: sportsbook backtest (for Kirtan's `eval/` lane)

**Status: prototype, not a final result. Written to show the idea works and
is worth building for real -- not to claim the current model beats Vegas.**

## The question

Josh's framing: if we'd bet the model's predictions through a real playoff
run, would we have made money? That's a genuinely good test for the
course's "measurable results" bar -- comparing against an actual market is
a stronger signal than comparing against raw win/loss, because games have
upsets a market has already priced in.

This isn't a new idea for the project: `agent/tools.py`'s
`retrieve_betting_line` TODO already cites Sadovnik's 7/07 framing almost
verbatim, and names Kirtan as owner. This prototype is a demonstration that
the idea is buildable, not a claim on that ownership.

## What exists to build it

`data/raw/odds/primary/nba_2008-2026.csv` -- 24,441 rows, one per game,
2008-2026, with spread, total, moneyline (both sides), and final scores.
This was already sitting on disk locally (dated 2026-07-13, before this
session), gitignored under the `data/raw/` policy, and is **not yet part of
the shared data layer**. If the team wants to adopt it, that's a deliberate
`data/raw/odds/` commit for Patrick/Kirtan to review -- I didn't add it to
git as part of this proposal.

**Important coverage gap, found by running this, not assumed:** the
moneyline columns are fully populated through the 2022 season, partially
populated for 2023, and **empty for 2024, 2025, and 2026** -- so this file
cannot backtest last season or the upcoming one. It's a real dataset for
*validating the approach*, not for testing on recent data. Someone will
need a live or more recent odds source (many exist, several free) to
backtest anything from the last two seasons.

## The prototype: `proposals/sportsbook_backtest.py`

Runnable now:

```
python proposals/sportsbook_backtest.py --season 2022 --playoffs-only
python proposals/sportsbook_backtest.py --season 2022
python proposals/sportsbook_backtest.py --season 2019 --playoffs-only
```

It imports and calls the **real, unmodified** `predict_win_probability`
stub from `agent/tools.py` -- it does not reimplement or fake the
prediction. For each game: convert the market's moneyline to an implied
probability, compare to the model's probability, and if the edge clears a
threshold (default 5 points), place a flat 1-unit bet. Reports accuracy,
Brier score (calibration), bet count, and ROI, against a baseline of
"always back the market favorite."

## Actual results (real runs, not illustrative)

| Run | Games | Accuracy | Brier | Edge bets | Edge ROI | Baseline (always-favorite) ROI |
|---|---|---|---|---|---|---|
| 2022 playoffs | 87 | 51.7% | 0.257 | 68 | **-4.3%** | -9.3% |
| 2022 full season | 1323 | 60.0% | 0.239 | 973 | **-5.3%** | -3.8% |
| 2019 playoffs | 82 | 59.8% | 0.229 | 66 | **+22.0%** | -8.6% |

## Honest read of these numbers

- Raw win/loss accuracy (51-60%) is unsurprising -- net rating alone
  predicts the better team reasonably often. It says nothing about beating
  a market that already knows the teams are unevenly matched.
- The **edge-betting ROI is negative in 2 of 3 runs**, meaning the current
  placeholder does not have a real, exploitable edge over the market. The
  positive 2019 playoffs number is very likely noise from a 66-bet sample,
  not a discovered edge -- one good stretch is expected by chance and
  should not be reported as "the model beats Vegas" without a much larger
  sample and out-of-sample testing.
- This is exactly what should be expected from a heuristic that (per its
  own docstring) **ignores the injury list entirely** and, for real data,
  has no rest signal wired in either (see the weighting proposal). The
  market prices in information this stub doesn't have access to yet.
- The value here isn't "we beat the sportsbook" -- it's that **the
  measurement tool itself works**, is honest about a real data gap
  (2024-26 moneylines missing), and is ready to point at a stronger model
  the moment one exists. That's the actual "measurable results" deliverable
  the course wants, and it's reusable indefinitely, including on next
  season once real odds and real outcomes exist for it.

## What I'm proposing to the team, not deciding unilaterally

1. Fold a version of this into `eval/` once we agree on scope -- Kirtan
   owns the replay harness / ablation runner per the README, so it should
   land there under his design, not get merged from `proposals/` as-is.
2. Source odds data that actually covers 2024-25 and 2025-26 before trying
   to backtest anything recent.
3. Re-run this exact script once `predict_win_probability` is Sarvesh's
   real model instead of the net-rating stub -- that's the test that
   actually matters for the "measurable results" rubric bar.
