---
tool: predict_win_probability
use_when: After matchup context. Adds Model A output as one additional data point for Model C.
---

## What it gives you

The exact Model A `home_win_prob` shown by the UI, plus the pregame feature values,
training seasons, and holdout accuracy recorded with the fitted model.

## When to call it

Every run, after `retrieve_matchup_context`.

## How to read it

It is a logistic regression over rolling point margin, win percentage, rest,
back-to-backs, injury load, and games played. The fitted weights use only the two
training seasons. Live feature inputs stop at the requested cutoff. This tool and
the UI's Model A button call the same function.

## Rules

- Treat `home_win_prob` as peer evidence alongside the matchup, team-form, injury,
  and rest outputs. It is one additional data point, not the starting answer.
- Synthesize the final probability from the complete gated evidence. Model C may
  agree or disagree with Model A.
- State a material agreement or disagreement and the evidence behind it in
  `key_factors`.
- If it returns an error or unavailable status, report that in `missing`. Do not
  substitute your own version of Model A.
