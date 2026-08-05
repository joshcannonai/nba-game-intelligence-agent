---
tool: predict_win_probability
use_when: Always. This is the number the report is built around.
---

## What it gives you

`home_win_prob` from a logistic regression trained on 2023-24 and 2024-25, plus the
eight features it used and its holdout accuracy.

## When to call it

Every run, after `retrieve_matchup_context`.

## How to read it

It scored **66.5%** on all 1,322 games of 2025-26, a season it never trained on,
against 55.5% for always picking the home team and 69.0% for the Vegas closing line.
It sees form, win percentage, rest, back-to-backs and injury load — nothing your
retrieval tools cannot also show you.

## Rules

- **Treat its number as the answer unless you have a concrete, specific reason it is
  wrong** — and if you move off it, say exactly why in `key_factors`.
- This is not deference for its own sake. We measured it. Across two paired 40-game
  samples the agent overruled the model 19 times and was wrong on 15 of them
  (two-sided sign test p ≈ 0.019). Overruling made the prediction worse, reliably.
- Good reasons to move off it: a tool returned something the model provably cannot
  see. Bad reasons: the injury list looks alarming (see
  `skills/retrieve_injuries.md`), the favourite "feels" wrong, or the probability
  seems too confident.
- If it returns `awaiting_input`, the model file is missing. Say so. Do not
  substitute your own estimate.
