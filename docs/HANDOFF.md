# Session handoff — 2026-08-03

State of the agent lane, for whoever picks this up next (a teammate, the advisor,
or an agent joining cold).

**Rough draft of the final report is due 2026-08-04, ~3:00 PM, before class.**
Semester ends ~2026-08-11.

---

## The one-paragraph version

The system predicts NBA games without ever seeing the future, and that claim is
checkable three ways: a fitted model (66.5% on a season it never trained on,
against 55.5% for always picking home and 69.0% for Vegas), an agent calling
seven date-gated tools, and a harness that scores them against each other. The
headline finding was negative — handing the agent the model's number made it
*worse*, because it overruled the model and was wrong 15 times out of 19. We then
fixed it: each tool now carries a written rule set, and on the same 40 games the
agent's accuracy went from 55.0% to 65.0%.

## From the group call, 2026-08-03

| Commitment | Owner | Status |
|---|---|---|
| Local LLM that calls the database tools | Josh | **done** — `--model ollama`, ~40s/game, no API key |
| Build a "skill" per tool, send the doc to Patrick | Josh | **done** — `skills/`, doc below |
| Review/edit the skills doc | Patrick | waiting on him |
| Get the draft report as complete as possible | Patrick | draft is written; §8 and §12 are his and Sarvesh's |
| Review the draft | Sarvesh | waiting |
| Push work to GitHub, compare against Sarvesh's model | Patrick | by Tuesday |

Also decided on that call: progress gets demonstrated to the professor via
**Colab output, not a UI**. Sarvesh's model shows itself that way. Ours can be
shown with `python -m agent.run --model ollama ...` and `python -m eval.three_arms`.

**Skills review doc (send this to Patrick):**
`https://docs.google.com/document/d/1klXHmPwWRK2ogE6mVRTz7rV-pk_Wu1dpVyUHsfjwtd4/edit`
— currently private. Needs link-sharing turned on before he can open it.

**Report draft (already circulated):**
`https://docs.google.com/document/d/1kr87GpueIulw8N_Fo4lmUSEs7ju0XdH_8dYRlzckv7w/edit`
— regenerate in place with `./scripts/to_gdoc.sh`. Never create a new one; that
link is already out.

## Where things stand

| | state |
|---|---|
| tests | **73 passing** |
| tools | **7**, each with a skill file · 5 return real data · 2 awaiting input |
| model | logistic regression, trained 2023-24 + 2024-25, tested on 2025-26 |
| gating | two independent gates — query-time filters *and* an on-disk snapshot |
| local agent | works end to end on Gemma 4 via Ollama, no API key |

### What runs today

```bash
python -m models.train                                # fit the model, ~7s
python -m eval.three_arms                             # arm A + baselines, ~3s
python -m eval.injury_impact                          # the top-scorer measurement
python -m agent.run --status --source real            # tool inventory, instant
python -m agent.run --model ollama --source real \
    --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30   # the real agent loop, ~40s
python -m scripts.gate_snapshot --as-of 2026-01-14    # materialise the gate
python scripts/skills_doc.py                          # regenerate docs/SKILLS.md
streamlit run ui/app.py                               # 4-tab report UI
streamlit run ui/chat.py                              # ask it questions
```

`--model ollama` needs `ollama serve` up and `ollama pull gemma4` once.
Streamlit has no auto-reload here — **restart the server after any code change**.

---

## Shipped 2026-08-03

### The skills layer

Each tool has a Markdown file in `skills/` saying when to call it and what to do
with the answer. The agent loads them into its system prompt at startup, so
editing one changes behaviour with no code change. `agent/skills.py` composes the
block from the tools the agent was actually given, which keeps arms B and C
differing by exactly one tool — `tests/test_skills.py` asserts it.

This came out of the 08-03 call, where the alternative was to dump everything in
a database and let the agent query freely. Rejected for the same reason the
betting-line tool had to go: a free-form query surface is how an agent reaches
data nobody intended it to see, and the date gate depends on every read passing
through `agent/sources.py`.

### The rule we could not write

The call agreed to encode "a >20 ppg player is out, so drop the odds by N%".
There is no N the data supports, and `eval/injury_impact.py` shows why:

| Comparison | Result |
|---|---|
| Naive, all games | home wins *less* when the **away** star is out — backwards |
| Restricted to teams that have a star | teams win **+5.6%** *more* without him (z = +2.6) |
| **Each team against itself** | **+0.0%** (se 3.3%, n = 21 teams) |

The first two are confounded: having a 20 ppg scorer is a property of good teams,
so the split compares strong rosters to weak ones. Only the within-team
comparison controls for that, and it finds nothing, with a spread from −32% to
+36%. So `skills/retrieve_injuries.md` tells the agent to report the injury list
and let the model price it.

**Caveat:** star-to-team mapping is from the prior season, so a player who changed
teams still counts against his old club. BOS shows 67 of 89 games "without star",
which is a departure, not an absence.

### It worked

Same 40 games (seed 0), nothing changed but the prompt:

| | arm A | arm C before | arm C after |
|---|---|---|---|
| accuracy | 75.0% | 55.0% | **65.0%** |
| log loss | 0.578 | 0.675 | **0.596** |
| Brier | 0.197 | 0.241 | **0.205** |
| overrides | — | 12 | **8** |

n = 40, one sample, so directional. The override sign test is now under-powered
(8 overrides, 2 successes, p ≈ 0.29) — that is *less overruling to measure*, not
a solved problem. **Do not report this as "fixed".**

### Also

- `docs/REPORT.md` rewritten: plainer prose, ~570 words shorter, cited data
  sources (§4) and a References appendix.
- `eval/crosscheck_odds.py` restored — it existed only on `josh/week5-eval-harness`
  and was dropped in the branch move, while HANDOFF cited its result. 9/10 reproduces.
- `agent/run.py` and `ui/app.py` were hard-coding "this is a stub, not the XGBoost
  model" while the fitted model was doing the work. Both now report real provenance.

---

## Next, in order

1. **Land PR #17, then #18.** `main` is 21 commits behind and has no `models/`,
   no `eval/`, no snapshot gate. Anyone cloning `main` sees half a project. This
   is the single highest-value action available.
2. **Share the skills doc with Patrick** (link above, needs link-sharing on).
3. **Second seed for the skills result.** `--arms ac --sample 40 --seed 1 --model
   ollama`, about 25 minutes. Without it §9.7 is one sample.
4. **Record the odds file's upstream URL** (Kirtan). It is the basis of the entire
   Vegas baseline and the repo does not say where it came from.
5. **Sarvesh: §8 of the report**, and resolve whether the XGBoost classifier is
   configured as a classifier. Flagged on 07-28, still open.
6. **Patrick: §12 and the data-pipeline section**; commit `season_schedule_2026.csv`
   to unblock `retrieve_schedule`.
7. **Fix the star-to-team mapping** with a current-season roster, which would
   strengthen both `eval/injury_impact.py` and the model's injury feature.

---

## Traps

- **The odds file keeps `score_away`/`score_home` in the same row as the line.**
  The single most likely way this project leaks. `odds_only.csv` is built without
  score columns and tests assert it.
- **Betting line is evaluation-only.** The agent has no tool for it and a test
  keeps it that way. Do not re-add one.
- **Model-knowledge gate only holds for 2025-26.** Gemma 4's cutoff is ~Jan 2025,
  so 2023-24 and 2024-25 are demos of the mechanism, not valid LLM eval games.
- **`importance` is `None`, not `0.0`** for players with no prior season.
- **The UI report tab is not agentic.** It runs `dry_run`.
- **Feature order in `win_probability.json` is positional.** Append only.
- **`langchain` must be 1.x.** The repo uses `create_agent`.
- **Skills change agent behaviour.** An edit to `skills/*.md` is a behaviour
  change with no code diff. Re-run the eval after any substantive edit.

## Source material

- Group call 2026-08-03 —
  `~/Cortex/LifeOS/Meetings/2026-08-03_sports-prediction-project-sync.md`
- Advisor meeting 2026-07-28 —
  `~/Cortex/LifeOS/Meetings/2026-07-28_nba-prediction-project-progress-review.md`
- Advisor meeting 2026-07-21 (betting line as baseline, not feature)
- PDP review 2026-07-07 (date-gated retrieval, role split)
