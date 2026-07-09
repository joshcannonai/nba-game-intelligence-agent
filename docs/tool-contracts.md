# Tool contracts (agent interface)

These are the function shapes the agent expects. Patrick / Kirtan (data) and Sarvvesh (models) can implement the real versions behind the same names and argument lists. Until then, `agent/tools.py` serves mock stubs with the same signatures.

All retrieval tools take an `as_of_date` (`YYYY-MM-DD`). Real implementations must return only records published on or before that date (date-gated retrieval, per the 2026-07-07 class decision).

## Data tools (Patrick + Kirtan)

### `retrieve_matchup_context(matchup_id: str, as_of_date: str) -> str`
JSON string with:
- `as_of_date`, `matchup_id`, `game_date`
- `home_team` / `away_team`: `abbr`, `name`, `record`, `off_rating`, `def_rating`, `pace`
- `rest`: `home_days_rest`, `away_days_rest`, `away_back_to_back` (bool)
- `injuries`: list of `{team, player, status, note, published}` filtered to `published <= as_of_date`
- `h2h_last_5`: list of `{date, winner, score}` filtered to `date <= as_of_date`

### `retrieve_player_splits(player_name: str, back_to_back: bool = False) -> str`
JSON string with season averages (`pts_avg`, `reb_avg`, `ast_avg`). When `back_to_back=True`, also include `b2b_pts_avg` (and any other fatigue splits you add later).

Suggested follow-ons (not stubbed yet):
- `retrieve_espn(as_of_date: str, ...)`
- `retrieve_nba_stats(as_of_date: str, ...)`

One function per source is fine, as long as the agent can call them as tools.

## Model tools (Sarvvesh)

### `predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str`
JSON string with:
- `model` (name/version)
- `as_of_date`, `home`, `away`
- `home_win_prob`, `away_win_prob`

Later: `predict_stat_line(...)` for the regression models.

## Agent (Josh)

- LangChain tool-calling loop in `agent/run.py`
- Build mode: Anthropic API (personal credits)
- Replay / production mode: local Ollama model with a known knowledge cutoff (wired in a later week)
- Dry run with no API key: `python -m agent.run --dry-run`

## How to plug in

1. Keep the tool name and arguments stable.
2. Replace the mock body in `agent/tools.py`, or import your real function and wrap it with `@tool`.
3. Return JSON strings (LangChain tools in this scaffold speak in strings).
