# Session handoff — 2026-07-21

State of the agent lane at the end of the day, for whoever picks this up next
(future Josh, a teammate, or an agent joining cold).

Semester ends **~2026-08-11**. Three weeks. The advisor asked for a working v1
within two.

---

## Where things stand

| | state |
|---|---|
| `main` | `1618a72` — green, 24 tests |
| open work | PR **#13** (draft) — `josh/week5-eval-harness`, 27 tests |
| unmerged, older | PR **#6** (draft, stale since 07-14) — the betting-line join |
| tools | 10 written and callable · 3 return real data · 1 placeholder logic · 6 awaiting input |

### What runs today

```bash
python -m agent.run --status --source real            # tool inventory, instant
python -m agent.run --dry-run --source real \
    --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24   # report, no LLM, instant
python -m agent.run --model ollama --source real ...  # the real agent loop, ~30s
python -m eval.replay --playoffs                      # the experiment
streamlit run ui/app.py                               # 4-tab UI, deterministic path
```

`--model ollama` needs `ollama serve` up. Streamlit has no auto-reload here
(watchdog isn't installed) — **restart the server after any code change**, or you
get new `app.py` against cached modules and silently wrong output.

---

## Shipped 2026-07-21

- **`--status` board** — probes all ten tools, prints built / placeholder /
  awaiting-input plus who owes each input. Deterministic, free.
- **Honest `missing`** in every path. The agent used to report `"missing": []`
  on real data because it only called the two tools it was required to call.
- **Streamlit UI** (`ui/app.py`) — report, tool inventory, gating proof, build
  status. Runs the deterministic path; no LLM anywhere in it.
- **Eval harness** (`eval/replay.py`) — the thing the PDP calls our primary
  contribution. Was an empty folder this morning.
- **2025-26 test set** — 1,322 games, 85 playoff, split so the line file carries
  no scores.
- **Injury weighting** — `player_importance()`; win probability now moves with
  the injury list instead of ignoring it.
- **`conftest.py` + `pytest.ini`** — bare `pytest` (what the README says to run)
  used to fail on a fresh clone with `ModuleNotFoundError`.

### Current numbers (placeholder heuristic, not XGBoost)

| | accuracy | log loss | Brier |
|---|---|---|---|
| season, stub | 59.5% | 0.676 | 0.240 |
| season, always-home | 55.5% | 0.687 | 0.247 |
| season, Vegas | **69.0%** | **0.578** | **0.198** |
| playoffs, stub | 58.8% | 0.725 | 0.260 |
| playoffs, Vegas | 58.8% | **0.656** | **0.234** |

The playoff accuracy tie with Vegas is **small-sample noise on 85 games** — don't
quote it as a result. The calibration gap (log loss) is the real signal.

---

## Next, in order

1. **Rework the UI flow to the advisor's architecture.** He wants "go" to first
   run Kirtan's gating script to materialise a local gated snapshot directory,
   then point the agent at *only* that directory. Today gating happens at query
   time inside `sources.py`. His version is provably airtight — the agent can't
   see ungated data because it isn't on disk. This is the biggest open item.
2. **Pull the newer injury data.** The log ends **2025-01-12**, so injury
   weighting contributes exactly nothing on the 2025-26 test set. Patrick pushed
   newer injury/betting data; wire it in.
3. **Land PR #13**, then **PR #6** (or close #6 — its odds join is partly
   superseded by `scripts/build_2026_testset.py`).
4. **Agent-arm evaluation.** `eval/replay.py` currently scores the *tool*
   directly, which is the classifier arm. The agentic arm needs a subset run
   through `run_matchup` (~30s/game, so sample ~30 games, not 1,322).
5. **Sit with Sarvvesh.** XGBoost drops into `predict_win_probability`'s existing
   signature; the harness picks it up with no other change.

---

## Traps

- **The odds file keeps `score_away`/`score_home` in the same row as the line.**
  This is the single most likely way this project leaks. `odds_2026.csv` is built
  without score columns and tests assert it — keep it that way, and tell Kirtan,
  whose gating tool emits the same data.
- **Betting line is evaluation-only** (advisor, 07-21). It must not feed the
  prediction or the system reads the answer off the market.
- **Model-knowledge gate only holds for 2025-26.** Gemma 4's cutoff is ~Jan 2025,
  verified behaviourally (knows the 2024 Finals, not 2025 or 2026). Every game in
  `game_logs_2024/2025.csv` predates that cutoff, so **those seasons are demos of
  the mechanism, not valid evaluation games.**
- **`importance` is `None`, not `0.0`**, for players with no prior season. Sorting
  or summing it needs `or 0.0` — a rookie is unknown, not worthless.
- **The UI is not agentic.** It runs `dry_run`. Don't describe it as the agent.
- **The win probability is not a model.** It's `net_rating_diff + 2.5 + injury
  cost`. Don't call it ML until XGBoost lands.

---

## Open decisions

- Exact regular-season/playoff boundary. The team said **April 14**; the odds data
  shows the first 2026 playoff game on **2026-04-18**. Pin it before Kirtan filters.
- Whether `retrieve_news` and `predict_best_player` survive the scope cut. Both
  are proposed for removal; neither is started.
- Whether the agent arm runs on all 85 playoff games or a sample.

## Source material

- Advisor meeting 2026-07-21 —
  `~/Cortex/Primary_Projects/WitnessAI/engine/audio/transcripts/2026-07-21_142201_manual-1422.summary.md`
- Group sync 2026-07-21 — Gemini notes in Drive (WitnessAI's capture of this one
  was empty audio)
- PDP — `CECS 499 PDP - NBA Game Intelligence Agent.pdf`
