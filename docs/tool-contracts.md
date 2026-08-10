# Tool contracts (current)

The agent exposes seven tools from `agent/tools.py`. `--source mock` uses a
deterministic fixture for tests; `--source real` reads the repository datasets.
Both sources use the same function names, arguments, date gate, and response
semantics.

```bash
python -m agent.run --status --source real
python -m agent.run --dry-run --source real \
  --matchup CHI-ORL-2025-12-01 --as-of 2025-11-30
```

All seven tools are live in the committed project data. If a required model file
or dataset is absent, the tool returns an explicit `status: awaiting_input`
payload with a reason. It never substitutes zero, a guess, or a heuristic.

## Date-gating contract

Every read is limited to information observable at `as_of_date`:

1. Injury transactions are replayed only through the cutoff.
2. Team form, head-to-head results, and model features use completed games before
   the cutoff; the target game's result cannot enter its own prediction.
3. Schedule dates remain visible because the schedule is published in advance,
   while outcomes remain gated.
4. End-of-season aggregates are used only when they were already complete.

These rules are enforced in `agent/sources.py` and `models/features.py` and
covered by the date-gating, model-contract, and snapshot-gate tests.

## Retrieval tools

`matchup_id` uses `AWAY-HOME-YYYY-MM-DD`, for example
`CHI-ORL-2025-12-01`.

- `retrieve_matchup_context(matchup_id, as_of_date)` — teams, pregame records,
  ratings, rest, injuries, and gated head-to-head history.
- `retrieve_player_splits(player_name, back_to_back=False)` — season averages
  plus a back-to-back split when supported.
- `retrieve_schedule(as_of_date, days_ahead=1)` — league schedule for the
  requested forward window.
- `retrieve_team_form(team_abbr, as_of_date)` — rolling record and scoring form.
- `retrieve_injuries(team_abbr, as_of_date)` — injury state reconstructed at the
  cutoff.

## Model tools

- `predict_win_probability(home_abbr, away_abbr, as_of_date)` — delegates
  directly to the frozen canonical predictor in `models/predict.py`.
- `predict_stat_line(player_name, matchup_id, as_of_date)` — projected points,
  rebounds, and assists with validation errors; suppresses players already known
  to be out.

Model-only, Both/arm C, and the evaluation harness all use the same canonical
win-prediction payload. Agent-only intentionally withholds the predictor tool.
Missing model weights produce the same `awaiting_input` response everywhere the
predictor is exposed.

## Changing an implementation

Keep the public tool name, arguments, JSON response shape, and gate intact.
Replace the implementation behind that boundary, add contract coverage, then run:

```bash
pytest
python -m scripts.build_site_data --check
```
