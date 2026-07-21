# Tool contracts (agent interface)

These are the function shapes the agent expects. Patrick / Kirtan (data) and Sarvvesh (models) can implement the real versions behind the same names and argument lists. `agent/tools.py` serves them from a **source**, selected at runtime:

```
python -m agent.run --dry-run                                   # mock fixture
python -m agent.run --dry-run --source real \
    --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24             # real CSVs, date-gated
```

`--source mock` is the deterministic fixture (no data files needed, used by tests). `--source real` reads the datasets on `main`. Both honour the same gating rules, so swapping the data layer in never changes the agent.

All retrieval tools take an `as_of_date` (`YYYY-MM-DD`). Real implementations must return only records published on or before that date (date-gated retrieval, per the 2026-07-07 class decision).

## What "date-gated" means in practice

Three rules, all enforced in `agent/sources.py` and locked by `tests/test_date_gating.py` (18 tests):

1. **Injuries are replayed, not summarised.** The injury dataset is a transaction log (`Relinquished` = out, `Acquired` = back). We replay it forward and stop at `as_of_date`, so the agent sees who was known to be out *that morning*. Two corrections the raw log needs: it never records a player *leaving* the team, so a naive replay still had Kemba Walker injured for Boston in 2024 — we drop anyone whose most recent appearance is with another team, plus anyone out longer than `STALE_INJURY_DAYS` (240).

2. **Season stats are prior-season only.** `Team Summaries.csv` and `Player Per Game.csv` are **end-of-season aggregates** — a 2024-25 rating row already contains games played after any mid-season `as_of_date`. Using the in-progress season would leak. We serve the **prior completed season** and label it (`basis` field). Current-season as-of ratings need rolling computation over game logs; that is a data-layer task, not an agent one.

3. **Schedule ≠ results.** Rest and back-to-back come from *when* games are played. The NBA publishes its schedule in August, so that is knowable on any as-of date and is **not** gated. Game *outcomes* (H2H) are gated at `as_of_date`. Drawing this line wrong makes a game scouted seven weeks out report "53 days rest."

Anything that cannot be computed comes back `null` **with a reason** — never a zero, never a guess. The agent's system prompt tells it to report those gaps rather than fill them.

## Open data-layer gap

`retrieve_matchup_context` needs a game-by-game **schedule + results** table. The four datasets on `main` are all season aggregates and carry none. `scripts/fetch_game_logs.py` pulls a thin version via `nba_api` (`data/samples/game_logs_<season>.csv`, ~1,225 games/season) so the agent's shapes match live data — it is a **sample, not the data layer**. The real scrape (more seasons, injury reports with true publication timestamps via `nbainjuries`) stays with Patrick + Kirtan.

## Data tools (Patrick + Kirtan)

`matchup_id` is `AWAY-HOME-YYYY-MM-DD` (e.g. `LAL-BOS-2024-12-25`).

### `retrieve_matchup_context(matchup_id: str, as_of_date: str) -> str`
JSON string with:
- `source` (`mock` | `real`), `as_of_date`, `matchup_id`, `game_date`
- `home_team` / `away_team`: `abbr`, `name`, `record`, `off_rating`, `def_rating`, `pace`, `basis` (which season the numbers are from)
- `rest`: `home_days_rest`, `away_days_rest`, `away_back_to_back`, `home_back_to_back` — or nulls + `unavailable` reason when no game logs
- `injuries`: list of `{team, player, status, note, published, days_out}` filtered to `published <= as_of_date`
- `h2h_last_5`: list of `{date, winner, score, home}` filtered to `date <= as_of_date`
- `ratings_basis`: why the ratings are prior-season
- `warnings` (optional): e.g. as_of falls past the end of the injury log

Raises `ValueError` if `as_of_date` is after tip-off — that would leak the result being predicted.

### `retrieve_player_splits(player_name: str, back_to_back: bool = False) -> str`
JSON string with season averages (`pts_avg`, `reb_avg`, `ast_avg`, `min_avg`, `games`, `basis`). When `back_to_back=True`, include `b2b_pts_avg` — or `null` plus `b2b_unavailable` if the source cannot compute a fatigue split (the real source cannot yet; that needs per-game logs).

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
- Build mode (`--model anthropic`, default): Anthropic API (personal credits)
- Replay / production mode (`--model ollama`): local Gemma 4 via Ollama, known
  knowledge cutoff, no API key -- `ollama pull gemma4` once
- Dry run with no API key or model call: `python -m agent.run --dry-run`

## How to plug in

1. Keep the tool name and arguments stable.
2. Replace the mock body in `agent/tools.py`, or import your real function and wrap it with `@tool`.
3. Return JSON strings (LangChain tools in this scaffold speak in strings).
