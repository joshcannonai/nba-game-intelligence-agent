# Session handoff — 2026-08-01

State of the agent lane, for whoever picks this up next (a teammate, the
advisor, or an agent joining cold).

Semester ends **~2026-08-11**. The advisor asked for a working v1; this is it.

---

## The one-paragraph version

The system predicts NBA games without ever seeing the future, and that claim is
now checkable three separate ways. There is a real model (66.5% on a season it
never trained on, against 55.5% for always picking the home team and 69.0% for
Vegas), a real agent calling seven date-gated tools, and a harness that scores
them against each other. The agent lost its betting-line tool along the way,
because it was caught quoting the market back at us as its own reasoning.

**The headline finding is a negative one, and it is the good kind.** We expected
the agent-plus-model arm to win. It lost badly: when the agent overruled the
model it was handed, it was wrong 15 times out of 19 (p ≈ 0.01, two samples pooled). The explanation
layer is currently costing us accuracy, and that is a far more interesting thing
to write up than a confirmed hypothesis would have been.

## Where things stand

| | state |
|---|---|
| tests | **64 passing** |
| tools | **7** written and callable · 5 return real data · 2 awaiting input |
| model | logistic regression, trained 2023-24 + 2024-25, tested on 2025-26 |
| gating | two independent gates — query-time filters *and* an on-disk snapshot |

### What runs today

```bash
python -m models.train                                # fit the model, ~7s
python -m eval.three_arms                             # arm A + baselines, ~3s
python -m eval.three_arms --arms abc --sample 40      # all three arms, ~50min
python -m agent.run --status --source real            # tool inventory, instant
python -m agent.run --dry-run --source real \
    --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24   # report, no LLM, instant
python -m agent.run --model ollama --source real ...  # the real agent loop, ~40s
python -m scripts.gate_snapshot --as-of 2026-01-14    # materialise the gate
streamlit run ui/app.py                               # 4-tab report UI
streamlit run ui/chat.py                              # ask it questions
```

`--model ollama` needs `ollama serve` up and `ollama pull gemma4` once.
Streamlit has no auto-reload here — **restart the server after any code change**,
or you get new `app.py` against cached modules and silently wrong output.

---

## Shipped 2026-08-01

### The model is real now

`models/` was an empty folder with a `.gitkeep`. It now holds a fitted model and,
more importantly, **an interface** — `predict(home, away, as_of)`. Three callers
depend on that signature and nothing else, so swapping in a different model is a
one-file change. See `models/README.md`, written for Sarvesh.

Trained on the two seasons *before* the one it is tested on. Split by season, not
by random shuffle: a random split lets a model learn from March to predict
January, which flatters accuracy by a few points that vanish under review.

| | accuracy | log loss | Brier |
|---|---|---|---|
| always pick home | 55.5% | 0.6871 | 0.2470 |
| **model (arm A)** | **66.5%** | **0.6118** | **0.2116** |
| Vegas closing line | 69.0% | 0.5782 | 0.1977 |

Train/test gap is **+0.3%** — not overfit. All 1,322 games of 2025-26.

Features come from `models/features.py`, which reuses the same gated accessors
the agent's tools use, so the model provably cannot see anything the agent
cannot. A test asserts the two implementations agree game for game.

### Ten tools became seven

| cut | why |
|---|---|
| `retrieve_news` | No source was ever found. Highest effort, least measurable payoff. |
| `predict_best_player` | Depended on `predict_stat_line`, which never started. A placeholder behind a placeholder. |
| `retrieve_betting_line` | **A leak, not a scope cut.** See below. |

Running the live agent on 2026-01-14 it wrote *"The closing betting line favors
the home team, ORL (-5.5 spread)"* straight into its key factors. The closing
line is the benchmark we grade ourselves **against**. An agent that reads the
market and repeats it scores well and has predicted nothing.

Telling the model not to peek is a request. Taking the tool away is a guarantee.
`agent.sources.closing_line` still exists and `eval/` still calls it directly, so
the Vegas baseline is untouched. `tests/test_date_gating.py` now asserts the tool
cannot come back.

### The three-arm experiment exists

`eval/three_arms.py`. Arms differ by **exactly one tool**, which a test enforces:

- **A** — model only, no LLM
- **B** — agent only, reasons from the retrieval tools
- **C** — agent + the model's number

Arm A and both baselines run over all 1,322 games in about three seconds. Arms B
and C call a language model once per game (~38s), so they run on a fixed random
sample, and the harness refuses a full-season LLM run rather than quietly
starting a 30-hour job. Every arm is scored on the *same* games — a paired
comparison on 40 beats an unpaired one on 1,322.

The harness prints one standard error next to the headline gap, because at n=40
that band is roughly ±8% and wider than the effect we are looking for.

### The result: the hypothesis was wrong

We predicted **C > A** — that an agent given the model's number would beat the
model alone. The opposite happened. Forty paired games, every arm scored on the
same games:

| arm | accuracy | log loss | Brier |
|---|---|---|---|
| **A — model only** | **75.0%** | **0.5782** | **0.1965** |
| B — agent only | 57.5% | 0.6577 | 0.2323 |
| C — agent + model | 55.0% | 0.6754 | 0.2409 |
| Vegas | 57.5% | 0.6488 | 0.2278 |
| always-home | 50.0% | 0.6982 | 0.2525 |

**C is 20 points worse than A**, outside one standard error. Handing the agent
the model's answer made it *worse* than the answer it was handed.

The headline accuracies are not the trustworthy part — arm A scores 75% on this
sample against 66.5% across the full season, so these 40 games happen to suit it.
The robust finding is paired, and immune to that:

```
agreed on          28
overruled on       12
  model was right  10
  agent was right   2
overruling helped 2/12 times -- p = 0.019
```

**When the agent overruled the model, it was wrong 10 times out of 12.**

Confirmed on a second sample (`--seed 1`), and the confirmation matters because
it also tempers the claim:

| sample | overrides | model right | agent right | p |
|---|---|---|---|---|
| seed 0 | 12 | 10 | 2 | 0.019 |
| seed 1 | 7 | 5 | 2 | 0.227 |
| **pooled** | **19** | **15** | **4** | **0.0096** |

Seed 1 on its own is **not** significant — only seven overrides, and a 5-2 split
is ordinary luck. The direction replicates (the agent's overrides succeed 17%
and 29% of the time, both far below the 50% you would get from a coin), and
pooled across 19 overrides it holds at p ≈ 0.01. Report the pooled number, and
say out loud that one 40-game sample could not have established this alone.

Seed 1 is also a good reminder of how noisy n=40 is: Vegas scored 57.5% on the
first sample and 77.5% on the second, on the same season.

It was not overruling at the margins either — the biggest reversals took a
confident correct call and inverted it:

| game | model | agent | actual |
|---|---|---|---|
| CHI-ORL-2025-12-01 | 0.815 | 0.249 | home won |
| IND-PHI-2026-01-19 | 0.741 | 0.242 | home won |

This is the project's most interesting result and it should lead the report. The
agent is not adding noise around a good estimate; it is systematically talking
itself out of the model's confident correct calls, most likely by over-weighting
the injury list it can see in `retrieve_injuries`.

**Do not quote arm A's 75% as our accuracy.** The season-long number is 66.5%.

### Leakage is mutation-tested

Tests that cannot fail prove nothing, so each rule was broken on purpose:

| mutation | tests that caught it |
|---|---|
| accumulators advanced before the feature row was emitted | 3 |
| form window drifted out of sync with the agent's accessor | 1 |
| test season added to the training seasons | 3 |

### Also

- Fixed `scripts/gate_snapshot.py --out` crashing for any path outside the repo.
- `ui/chat.py` answers market questions by explaining *why* it cannot see the
  line, instead of "nothing matched".

---

## Next, in order

1. **Chase down *why* the agent overrules the model.** This is now the most
   valuable open question in the project. The per-game reasoning is in the arm C
   output — read the `key_factors` on CHI-ORL-2025-12-01 and IND-PHI-2026-01-19
   and find out what made it abandon a 0.8 prediction. Best guess: it reads the
   injury list and over-weights it, the same failure the old heuristic had.
2. **Re-run with a second sample seed before the result goes in the report.**
   `--seed 1 --sample 40`. The paired override finding should hold; if it does
   not, we learned that 12 overrides is too few to sign-test.
3. **Sarvesh: beat 66.5%.** `models/README.md` lists four concrete weaknesses.
   Opponent-adjusted strength of schedule is probably the biggest single win.
4. **Patrick: commit `season_schedule_2026.csv`.** Last thing blocking
   `retrieve_schedule`; `data/raw/` stopped being gitignored on 7/21.
5. **Kirtan: `eval/crosscheck_odds.py` confirmed the odds file is the closing
   line** (9 of 10 games closer to closing). That was an open assumption the
   entire Vegas baseline rested on — it belongs in the report.
6. **Land PR #17**, then close or rebase the stale #6 and #13.

---

## Traps

- **The odds file keeps `score_away`/`score_home` in the same row as the line.**
  The single most likely way this project leaks. `odds_only.csv` is built without
  score columns and tests assert it — keep it that way.
- **Betting line is evaluation-only** (advisor, 07-21). The agent no longer has a
  tool for it, and a test keeps it that way. Do not re-add one.
- **Model-knowledge gate only holds for 2025-26.** Gemma 4's cutoff is ~Jan 2025,
  verified behaviourally. Every game in `game_logs_2024/2025.csv` predates that
  cutoff, so **those seasons are demos of the mechanism, not valid evaluation
  games for the LLM arms.** They are fine for training the model, which has no
  world knowledge to leak.
- **`importance` is `None`, not `0.0`**, for players with no prior season. Sorting
  or summing it needs `or 0.0` — a rookie is unknown, not worthless.
- **The UI report tab is not agentic.** It runs `dry_run`. `ui/chat.py` can be
  agentic if you switch the backend in the sidebar.
- **Feature order in `win_probability.json` is positional.** Reordering
  `FEATURE_NAMES` silently remaps every weight. Append only; a test checks it.
- **`langchain` must be 1.x.** The repo uses `create_agent`, which does not exist
  in 0.3. A 0.3 install with a 1.x `langchain-core` fails to import at all.

---

## Open decisions

- Exact regular-season/playoff boundary. The team said **April 14**; the odds
  data shows the first 2026 playoff game on **2026-04-18**.
- Sample size for arms B and C. 40 games is ±8% — enough to see a large effect,
  not a small one. More games is hours, not minutes.
- Whether `predict_stat_line` survives to the end of the semester or joins the
  cut list. Nothing has started behind it.

## Source material

- Advisor meeting 2026-07-21 —
  `~/Cortex/Primary_Projects/WitnessAI/engine/audio/transcripts/2026-07-21_142201_manual-1422.summary.md`
- Group sync 2026-07-28 —
  `~/Cortex/Primary_Projects/WitnessAI/engine/audio/transcripts/2026-07-28_144513_manual-1445.summary.md`
- PDP — `CECS 499 PDP - NBA Game Intelligence Agent.pdf`
