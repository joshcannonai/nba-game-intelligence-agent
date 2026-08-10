"""The tools the agent can call. This file IS the contract with the rest of the team.

Seven tools, down from ten. The three that went, and why -- because a scope cut
nobody can explain later just looks like work that was abandoned:

    retrieve_news         No source was ever found. Highest effort of the ten,
                          least measurable contribution. Cut on merit.
    predict_best_player   Was cut before predict_stat_line landed. The current
                          product reports individual player projections instead.
    retrieve_betting_line REMOVED FROM THE AGENT, kept for scoring. This one is
                          not scope, it is a leak. Watching the live agent run
                          on 2026-01-14, it pulled the closing line into its own
                          reasoning -- "the closing line favors ORL (-5.5)" --
                          and the closing line is the thing we grade ourselves
                          AGAINST. An agent that reads the market and repeats it
                          scores well and has predicted nothing. Telling the
                          model not to peek is a request; taking the tool away
                          is a guarantee. `agent.sources.closing_line` still
                          exists and eval/ still calls it directly.

Every function the agent needs in order to produce the report we described on 7/07
(who wins · a narrative · statistics) exists here NOW, with a stable name and a
stable signature.

EVERY tool in this file is written by the agent lane (Josh). Nobody else writes
agent code. What varies is whether the data or model each tool reads from exists
yet -- so a placeholder returns:

    {"status": "awaiting_input", "needs_from": "...", "needs": "..."}

`needs_from` names whoever produces that INPUT, not someone who owes a function.
Sarvesh trains the model; the tool that calls it is still mine. Patrick pulls the
schedule; the tool that reads it is still mine.

That is deliberate. A placeholder is not a stub that lies -- it tells the agent, in
plain terms, that the input does not exist yet and where it comes from. The agent
reports the gap instead of inventing an answer, so running it today prints an honest
status board of the whole project.

To finish one: keep the name and the arguments, replace the body once its input
lands. The agent never notices -- that is the entire point of the interface.

    inputs: data     (Patrick + Kirtan) -> feeds the retrieve_* tools
    inputs: models   (Sarvesh)         -> feeds the predict_* tools
    all tools + loop (Josh)             -> this file and agent/run.py
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.sources import get_source, parse_date, parse_matchup_id, player_is_out
from models.predict import predict
from models.predict_stat_line import model_available as stat_line_available
from models.predict_stat_line import predict_stat_line as run_stat_line


def _todo(tool_name: str, needs_from: str, needs: str, **ctx) -> str:
    """The honest placeholder. Never fabricates; names where the input comes from.

    The tool itself is mine either way -- `needs_from` is who produces the data or
    model it reads, not who owes a function.
    """
    return json.dumps(
        {
            "status": "awaiting_input",
            "tool": tool_name,
            "needs_from": needs_from,
            "needs": needs,
            **ctx,
            "note": (
                "This tool's input does not exist yet. Report it as unavailable in "
                "your output. Do NOT invent a value and do NOT treat it as zero."
            ),
        },
        indent=2,
    )


def build_tools(source, include_model: bool = True, without: tuple[str, ...] = ()):
    """Bind a data source into the tool set the agent gets.

    include_model=False withholds predict_win_probability. That is not a debug
    switch -- it is arm B of the three-arm experiment. Arm B has to reason its
    way to a winner from the retrieval tools alone; arm C gets the model's
    number handed to it. The difference between the two IS the measurement, so
    the two arms have to differ in exactly one tool and nothing else.

    `without` withholds tools by name, for the ablation study: run the same games
    with a tool removed and the accuracy change is that tool's contribution. It
    goes through the same path as include_model, so the skills block shrinks with
    it and the agent is never told about a tool it does not have.
    """

    # ---------------------------------------------------------------- WORKING

    @tool
    def retrieve_matchup_context(matchup_id: str, as_of_date: str) -> str:
        """Team ratings, rest, injuries and head-to-head for a matchup, as of a date.

        Only records published on or before as_of_date are returned. Anything that
        cannot be computed comes back null with a reason -- treat it as unknown,
        never as zero.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD, e.g. LAL-BOS-2024-12-25
            as_of_date: ISO date. Nothing published after this date is read.
        """
        return json.dumps(source.matchup_context(matchup_id, as_of_date), indent=2)

    @tool
    def retrieve_player_splits(player_name: str, back_to_back: bool = False) -> str:
        """A player's season averages, optionally their back-to-back (fatigue) split.

        Args:
            player_name: Full player name.
            back_to_back: If true, include the fatigue split when the source has one.
        """
        return json.dumps(source.player_splits(player_name, back_to_back), indent=2)

    # ------------------------------------------------- DATA LAYER (P + K) TODO

    @tool
    def retrieve_schedule(as_of_date: str, days_ahead: int = 1) -> str:
        """The games on the slate -- what the user picks from in the UI.

        The NBA publishes its schedule in August, so future GAME DATES are knowable
        on any as_of_date and are NOT leakage. Future RESULTS are.

        Args:
            as_of_date: ISO date the user is asking from.
            days_ahead: How many days of upcoming games to return.
        """
        return json.dumps(
            source.schedule(as_of_date, days_ahead=days_ahead), indent=2
        )

    @tool
    def retrieve_team_form(team_abbr: str, as_of_date: str, last_n: int = 10) -> str:
        """A team's CURRENT strength as of a date -- record and rating over recent games.

        Different from the season CSVs on main, which are END-OF-SEASON totals. Using
        those mid-season leaks the future, so today we fall back to the prior completed
        season, which is stale by December. This tool is the fix.

        Args:
            team_abbr: Team abbreviation, e.g. BOS.
            as_of_date: ISO date. Only games played before this date may be used.
            last_n: Window for rolling form.
        """
        if hasattr(source, "team_form"):
            form = source.team_form(team_abbr, as_of_date, last_n)
            if not form.get("unavailable"):
                return json.dumps(form, indent=2)
        return _todo(
            "retrieve_team_form",
            "Josh (built for 2025-26; other seasons need game logs)",
            "A rolling, as-of team rating from games played BEFORE as_of_date. Built "
            "from data/samples/game_logs_*.csv. Returns awaiting_input only when no "
            "in-season games exist yet (opening week) or that season's logs are "
            "absent -- then the caller uses prior-season ratings instead of guessing.",
            team_abbr=team_abbr,
            as_of_date=as_of_date,
            last_n=last_n,
        )

    @tool
    def retrieve_injuries(team_abbr: str, as_of_date: str) -> str:
        """Who was KNOWN to be out, on the morning of the game.

        Works today by replaying the injury transaction log and stopping at as_of_date.
        The log now runs to 2026-05-29, so the 2025-26 replay window is covered.

        Known limit: these are transaction dates -- when a player was placed on or
        activated from the injured list -- not the moment the news broke. An as-of
        query on the morning of a game can therefore see a same-day placement.

        Args:
            team_abbr: Team abbreviation.
            as_of_date: ISO date.
        """
        return json.dumps(source.injuries(team_abbr, as_of_date), indent=2)

    # ---------------------------------------------------------------- MODELS

    @tool
    def predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str:
        """Probability the home team wins, from a model fitted on prior seasons.

        Backed by models/predict.py -- a logistic regression trained on 2023-24 and
        2024-25 and never on the season being replayed. It reads form, rest,
        back-to-backs and injury load, all as of the morning of the game. Holdout
        accuracy on all 1,322 games of 2025-26 is 66.5%, against 55.5% for simply
        always picking the home team.

        If models/win_probability.json is missing, the canonical predictor returns
        an explicit ``awaiting_input`` envelope. It never substitutes a heuristic.

        Args:
            home_abbr: Home team abbreviation.
            away_abbr: Away team abbreviation.
            as_of_date: ISO date for the prediction.
        """
        return json.dumps(predict(home_abbr, away_abbr, as_of_date), indent=2)

    @tool
    def predict_stat_line(player_name: str, matchup_id: str, as_of_date: str) -> str:
        """Projected points / rebounds / assists for one player in this game.

        The "statistics" half of the report we pitched. Backed by
        models/predict_stat_line.py -- ridge regressions on a player's trailing
        5- and 10-game form, rest, and home/away splits, fitted on 2023-24 and
        validated on 2024-25. It is never fitted on the season being replayed.

        Returns `status: unavailable` when the gated injury list already marks the
        player out, or when observable pregame history is insufficient. Single-game
        lines are high variance; the payload carries the test mean absolute error so
        the size of the error is visible next to the number.

        Args:
            player_name: Full player name.
            matchup_id: AWAY-HOME-YYYY-MM-DD
            as_of_date: ISO date.
        """
        away, home, _ = parse_matchup_id(matchup_id)
        if player_is_out(player_name, (away, home), parse_date(as_of_date)):
            return json.dumps(
                {
                    "status": "unavailable",
                    "tool": "predict_stat_line",
                    "player": player_name,
                    "as_of_date": as_of_date,
                    "reason": (
                        f"{player_name} is listed Out as of {as_of_date}; "
                        "the projection is suppressed rather than implying "
                        "pregame participation."
                    ),
                },
                indent=2,
            )

        if not stat_line_available():
            return _todo(
                "predict_stat_line",
                "josh",
                "models/stat_line.json is missing. Run "
                "`python -m models.train_stat_line`.",
                player_name=player_name,
                matchup_id=matchup_id,
                as_of_date=as_of_date,
            )
        return json.dumps(
            run_stat_line(source, player_name, matchup_id, as_of_date), indent=2
        )

    tools = [
        retrieve_matchup_context,
        retrieve_player_splits,
        retrieve_schedule,
        retrieve_team_form,
        retrieve_injuries,
        predict_stat_line,
    ]
    if include_model:
        tools.append(predict_win_probability)
    if without:
        unknown = set(without) - {t.name for t in tools}
        if unknown:
            raise ValueError(f"cannot withhold unknown tools: {sorted(unknown)}")
        tools = [t for t in tools if t.name not in without]
    return tools


# Default tool set (mock) so `from agent.tools import TOOLS` still works.
TOOLS = build_tools(get_source("mock"))
