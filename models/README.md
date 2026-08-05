# models/ — the win-probability model

**Sarvesh: this is the handoff. There is a working model here already, so nothing is
blocked on you. Your job is to beat it, and swapping yours in is one file.**

---

## What is here now

A logistic regression trained on the 2023-24 and 2024-25 seasons and scored on
2025-26, which it never sees during training.

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **this model** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

*All 1,322 games of 2025-26. Train/test accuracy gap is +0.3%, so it is not overfit.*

Vegas is the bar, and we are 2.5 points off it. Closing that gap is the interesting
problem — the market has injury beat reporters and real money, and we have a laptop.

## Run it

```bash
python -m models.train          # refits, prints weights, rewrites the JSON (~7s)
python -m eval.three_arms       # scores arm A against both baselines (~3s)
```

## The three files

| file | what it does |
|---|---|
| `features.py` | Builds the as-of feature vector. **Read this one first.** |
| `train.py` | Fits the model, writes `win_probability.json`. |
| `predict.py` | `predict(home, away, as_of) -> dict`. Everything else calls this. |

`win_probability.json` is committed on purpose — it is a few hundred bytes of named
numbers rather than a pickle, so you can read the weights in a pull request and it
loads without sklearn.

## The interface — this is the contract

```python
from models.predict import predict

predict("BOS", "ORL", "2026-01-14")
# {"status": "ok", "home_win_prob": 0.6402, "away_win_prob": 0.3598, ...}
```

Three callers depend on this shape and nothing else:

- `eval/three_arms.py` — arm A
- `agent/tools.py` — hands it to the agent as arm C
- `ui/app.py` — one game at a time

**Keep `home_win_prob` as a float 0–1 and everything keeps working.**

## Swapping in your model

Easiest path, and the one that needs no changes anywhere else:

1. Train your XGBoost on the same features from `models/features.py`.
2. Write out the same JSON shape from `train.py` (or add a `models/train_xgb.py`).
3. Change `trained_by` to `"sarvesh"` so the UI shows whose model produced the number.

If XGBoost will not fit that shape — and it probably will not, since a tree
ensemble is not a list of coefficients — then instead:

1. Save your model however you like (`.joblib` is already gitignored; commit a
   loader script, not the binary).
2. Rewrite the body of `predict()` in `models/predict.py`.
3. Keep the signature and the returned keys identical.

Run `python -m pytest tests/test_model_contract.py` afterwards. Twelve tests check
the contract and the leakage rules; if they pass, the agent and the harness will
both pick your model up with no other edits.

## Two rules the tests enforce

**1. Never train on the test season.** `TRAIN_SEASONS = (2024, 2025)` and
`TEST_SEASON = 2026`. Split by season, not by a random shuffle — a random split lets
the model learn from March to predict January, which inflates accuracy by a few
points that vanish the moment anyone checks.

**2. A feature for a game may only use games before it.** The trap is the obvious
way to compute a season win percentage: group the season, take the mean, and you
have silently included the game you are predicting. `features.py` advances every
accumulator *after* emitting the row, never before.

Both rules are mutation-tested. We broke each one on purpose and confirmed tests fail:

| what we broke | tests that caught it |
|---|---|
| accumulators advanced before the row was emitted | 3 |
| form window drifted out of sync with the agent | 1 |
| test season added to the training seasons | 3 |

## Things worth improving

- `home_games_played` and `away_games_played` carry meaningful weight, which is a
  little suspicious — the two are near-identical for any given game, so the model
  may be fitting a schedule artifact. Worth testing whether dropping them hurts.
- The injury feature sums a prior-season minutes/points proxy and treats every
  listed player as fully out. It over-penalises.
- No opponent-adjusted strength: a 5-0 run against bad teams counts the same as
  5-0 against good ones. This is the strength-of-schedule idea, and it is probably
  the biggest single win available.
- Rest is capped naively — 2 days is assumed when a team has no prior game.
