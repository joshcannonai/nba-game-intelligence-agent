# Agent design notes

Plain-language walkthrough of `agent/` for presenting to the team and Sadovnik.
Written so any of this can be explained live, not just read off a slide.

## What it does today

Given a matchup (`LAL-BOS-2024-12-25`) and an as-of date, the CLI produces a
structured pregame report using only data that would have been knowable on
that date. Two ways to run it:

```
python -m agent.run --dry-run --source real \
    --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24
```

exercises the real tool contracts and date gating with no LLM call --
deterministic, free, and what `pytest` runs against. The full run is the
actual LangChain tool-calling loop, with a `--model` flag choosing the
backend:

```
python -m agent.run --source real --matchup ... --as-of ...              # Claude (build mode, default)
python -m agent.run --model ollama --source real --matchup ... --as-of ...  # local Gemma 4, no API key
```

`--model ollama` is the leakage-safe path for replay/production: Claude's
training cutoff isn't something we can pin to a date the way an open
model's release date is. Verified live, not just wired -- ran the full
loop against two real matchups on local Gemma 4 (`ollama pull gemma4`),
both producing valid structured reports pulling real data through the
tools.

## The honest-placeholder pattern

Every tool the final report needs exists now with a stable name and
signature (see `agent/tools.py`'s module docstring). Most bodies are
placeholders. A placeholder either returns real, gated data or an explicit

```json
{"status": "not_implemented", "owner": "...", "needs": "..."}
```

-- never a guess, never a silent zero. The point: running the agent against
real data today prints an honest status board of the whole team's progress,
not a demo that quietly fakes the 80% that isn't built. The agent's own
system prompt tells it to report gaps as `missing` in its final JSON instead
of papering over them.

## Why date-gating is the whole architecture

Three rules, all enforced in `agent/sources.py` and locked by
`tests/test_date_gating.py`:

1. **Injuries are replayed, not summarized.** The injury log is a
   transaction log (`Relinquished` / `Acquired`). Replaying it forward and
   stopping at `as_of_date` reproduces what a person could have known that
   morning. Two corrections the raw log needs: it never records a player
   *leaving* a team (a naive replay still had Kemba Walker "out" for Boston
   in 2024), and a relinquish older than 240 days with no return is an
   unrecorded departure, not an active injury.
2. **Season stats are prior-season only.** The season CSVs are
   end-of-season aggregates -- a 2024-25 rating row already contains games
   played after any mid-season as-of date. We serve the prior completed
   season instead and label it (`basis` field). True as-of ratings need
   rolling computation over game logs, which is the open data-layer gap.
3. **Schedule dates are not gated; results are.** Rest and back-to-back come
   from *when* games are played, and the NBA publishes its schedule in
   August -- so that's knowable on any as-of date. Game outcomes (H2H) are
   gated. Getting this backwards reports "53 days rest" for a game scouted
   seven weeks out.

Anything that can't be computed comes back `null` with a reason, never a
zero and never a guess.

## Current status (2026-07-15)

Verified by running the commands myself, not asserted from memory:

- **Working, tested:** `retrieve_matchup_context`, `retrieve_player_splits`,
  `retrieve_injuries` -- real data, `pytest -q` is 24 passed, `ruff check .`
  is clean. The `--dry-run --source real` CLI path runs end to end on a real
  matchup and prints a full report (ratings, rest, H2H, injuries, a
  probability, a narrative).
- **Placeholder, clearly labeled:** `predict_win_probability` -- a net-rating
  + rest heuristic (`stub_net_rating_v0`). It explicitly does not use the
  injury list yet. Sarvesh's XGBoost model replaces this body; the interface
  is what's locked, not the math.
- **Not built, each names its owner in code:** `retrieve_schedule`,
  `retrieve_team_form`, `retrieve_news`, `retrieve_betting_line` (data
  layer -- Patrick/Kirtan), `predict_stat_line`, `predict_best_player`
  (models -- Sarvesh).
- **Full LLM loop, now verified:** ran end to end on `--model ollama`
  (local Gemma 4, no API key) against two real matchups, both producing
  valid structured reports with real injuries/rest/ratings/H2H and no
  hallucinated stats. The Anthropic path (`--model anthropic`, still the
  default) is unchanged code but untested this session -- no personal
  `ANTHROPIC_API_KEY` set in `.env` on this machine. Both backends share
  the same tools and system prompt, so this isn't two different agents.

## What's proposed, not built, for Tuesday

Two ideas came up in review that are genuinely good for the course's
"measurable results" bar, but aren't my lane per the role split locked at
the 2026-07-07 PDP review (`data/` = Patrick+Kirtan, `models/` = Sarvesh,
`eval/` = Kirtan, `agent/` = me). Writing them as final code myself would
mean grading myself on my teammates' assignments and duplicating work they
haven't seen yet. Written up as proposals instead, for the team to fold in
or not:

- `proposals/weighting-scheme-proposal.md` -- for Sarvesh: how injuries,
  rest, and H2H could factor into `predict_win_probability` beyond net
  rating, and what the current stub is missing.
- `proposals/sportsbook-backtest-prototype.md` +
  `proposals/sportsbook_backtest.py` -- for Kirtan's `eval/` lane: a working
  prototype (using historical odds data already sitting locally,
  `data/raw/odds/`, ungitignored and not yet part of the shared data layer)
  testing whether following a model's predictions through a past season
  would have beaten the sportsbook line. Connects directly to the
  `retrieve_betting_line` tool that's already stubbed with Sadovnik's 7/07
  framing: beating the line is a better signal than beating the raw result,
  because games have upsets.

## AI-disclosure

Per the course's AI-use policy ("AI-assisted code is fine, but you own and
can explain every line you merge"): this document, the two proposal
artifacts above, and the `--model ollama` backend in `agent/run.py` were
written by Claude (Anthropic), at my direction, on top of the `agent/`
code that was already built and merged to `main` before this session
(PRs #7-9). **I have not yet read through any of this line by line** --
that review is the required next step before I present any of it, not an
afterthought. Nothing here should be presented or merged as final until
I've done that.
