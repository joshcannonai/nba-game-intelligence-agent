---
tool: retrieve_team_form
use_when: You need current strength rather than last season's.
---

## What it gives you

A rolling 10-game record and average point differential, using only games played
on or before `as_of_date`.

## When to call it

When `retrieve_matchup_context` reports `basis: prior completed season`, or whenever
you are about to make a claim about how good a team is *right now*.

## How to read it

`avg_point_diff` is the useful number. Roughly: +5 is a strong team, 0 is average,
−5 is weak. It is a better guide in December than a prior-season rating, which is
describing a roster that may no longer exist.

## Rules

- Under 5 games played, the window is noise. Say the sample is small rather than
  reporting the number as though it were settled.
- This is *not* opponent-adjusted. A 5–0 run against weak teams looks identical to
  5–0 against strong ones. If a team's record and its point differential disagree,
  trust the differential and say why.
- For Model B, compare both teams and treat the difference in rolling point
  differential as the primary strength signal. Use current win percentage only as
  corroboration. Start from the 55% home-team base rate, keep an otherwise even
  matchup near that base rate, and reserve 60-70% for cases where form margin and
  record agree strongly.
