"""The tools the agent can call. This file IS the contract with the rest of the team.

Vocabulary, because these words get used as if they were interchangeable:

    A TOOL is a Python function in this file. Deterministic. Date-gated. It
    returns JSON. It has no opinion about who wins.

    THE AGENT is the LLM loop in agent/run.py. It chooses which tools to call
    and writes the report. It has no CSV access, no web, and no second path
    around the gate. If a fact is not in a tool result, the agent does not
    have it.

    THE PREDICTOR (Model A) is a logistic regression. It is not an agent. It
    is not a tool except when Model C is allowed to call predict_win_probability,
    which wraps the same function POST /api/predict uses.

Seven tools. retrieve_schedule is not one of them: the UI already loads the
game list from the CSV, and rest/B2B already live on retrieve_matchup_context.
Listing other games on the slate does not help a bound one-game prediction.
The three that were cut, and why:

    retrieve_news         No source with reliable publication timestamps.
    retrieve_schedule     Slate picker, not a prediction input. Forbidden so
                          it cannot come back as a no-op tool.
    retrieve_betting_line REMOVED FROM THE AGENT, kept for scoring. Watching
                          the live agent on 2026-01-14, it pulled the closing
                          line into its own reasoning. Telling the model not
                          to peek is a request; taking the tool away is a
                          guarantee.

predict_best_player is back. It ranks the gated rotation with predict_stat_line
and returns the highest projected points. It is optional on the winner path
(same rule as predict_stat_line) and exists so a player question has a real
answer instead of a placeholder behind a placeholder.

`needs_from` names whoever produces that INPUT, not someone who owes a function.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.sources import get_source, parse_date, parse_matchup_id, player_is_out
from models.predict import predict as predict_win_model
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


def build_tools(
    source,
    include_model: bool = True,
    without: tuple[str, ...] = (),
    *,
    required_as_of_date: str | None = None,
    required_matchup_id: str | None = None,
):
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

    authorized_teams: set[str] | None = None
    if required_matchup_id is not None:
        away, home, game_date = parse_matchup_id(required_matchup_id)
        authorized_teams = {away, home}
        if (
            required_as_of_date is not None
            and parse_date(required_as_of_date) >= game_date
        ):
            raise ValueError("authorized as_of_date must be before the game date")

    def reject_unbound_request(
        tool_name: str,
        *,
        as_of_date: str | None = None,
        matchup_id: str | None = None,
        team_abbr: str | None = None,
    ) -> str | None:
        """Stop an agent from changing the server-authorized evaluation request.

        The UI binds every run to one matchup and one cutoff before the model is
        invoked. A model-selected argument is untrusted: returning this error
        before touching ``source`` guarantees that a later date or different game
        cannot reach the context window, even transiently.
        """
        problems = []
        if required_as_of_date is not None and as_of_date != required_as_of_date:
            problems.append(
                f"as_of_date must equal the authorized cutoff {required_as_of_date}"
            )
        if (
            required_matchup_id is not None
            and matchup_id is not None
            and matchup_id != required_matchup_id
        ):
            problems.append(
                f"matchup_id must equal the authorized matchup {required_matchup_id}"
            )
        if (
            team_abbr is not None
            and authorized_teams is not None
            and team_abbr not in authorized_teams
        ):
            problems.append(
                f"team_abbr must be one of the authorized teams {sorted(authorized_teams)}"
            )
        if not problems:
            return None
        return json.dumps(
            {
                "status": "error",
                "tool": tool_name,
                "error": "; ".join(problems),
                "requested_as_of_date": as_of_date,
                "requested_matchup_id": matchup_id,
                "requested_team_abbr": team_abbr,
                "data_source_read": False,
            },
            indent=2,
        )

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
        rejected = reject_unbound_request(
            "retrieve_matchup_context",
            as_of_date=as_of_date,
            matchup_id=matchup_id,
        )
        if rejected:
            return rejected
        return json.dumps(source.matchup_context(matchup_id, as_of_date), indent=2)

    @tool
    def retrieve_player_splits(
        player_name: str, as_of_date: str, back_to_back: bool = False
    ) -> str:
        """A player's season averages, optionally their back-to-back (fatigue) split.

        Args:
            player_name: Full player name.
            as_of_date: ISO cutoff used to select the prior completed season.
            back_to_back: If true, include the fatigue split when the source has one.
        """
        rejected = reject_unbound_request(
            "retrieve_player_splits", as_of_date=as_of_date
        )
        if rejected:
            return rejected
        return json.dumps(
            source.player_splits(
                player_name,
                as_of_date=as_of_date,
                back_to_back=back_to_back,
            ),
            indent=2,
        )

    @tool
    def retrieve_team_form(team_abbr: str, as_of_date: str, last_n: int = 10) -> str:
        """A team's CURRENT strength as of a date -- record and rating over recent games.

        Different from the season CSVs on main, which are END-OF-SEASON totals. Using
        those mid-season leaks the future, so today we fall back to the prior completed
        season, which is stale by December. This tool is the fix.

        Args:
            team_abbr: Team abbreviation, e.g. BOS.
            as_of_date: ISO date. Only games played on or before this date may be used.
            last_n: Window for rolling form.
        """
        rejected = reject_unbound_request(
            "retrieve_team_form",
            as_of_date=as_of_date,
            team_abbr=team_abbr,
        )
        if rejected:
            return rejected
        if hasattr(source, "team_form"):
            form = source.team_form(team_abbr, as_of_date, last_n)
            if not form.get("unavailable"):
                return json.dumps(form, indent=2)
        return _todo(
            "retrieve_team_form",
            "Josh (built for 2025-26; other seasons need game logs)",
            "A rolling, as-of team rating from games played THROUGH as_of_date. Built "
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

        Gated the same way every other historical read is: the transaction log
        is replayed and stopped at as_of_date. The JSON says so (`gated`,
        `knowledge_cutoff`, `injury_gate`). Later rows are unread.

        Known limit: these are transaction dates -- when a player was placed on
        or activated from the injured list -- not the moment the news broke.

        Args:
            team_abbr: Team abbreviation.
            as_of_date: ISO date. Transaction Date must be <= this cutoff.
        """
        rejected = reject_unbound_request(
            "retrieve_injuries",
            as_of_date=as_of_date,
            team_abbr=team_abbr,
        )
        if rejected:
            return rejected
        return json.dumps(source.injuries(team_abbr, as_of_date), indent=2)

    # ---------------------------------------------------------------- MODELS

    @tool
    def predict_win_probability(matchup_id: str, as_of_date: str) -> str:
        """The same Model A win probability shown by the UI.

        Backed by the committed logistic model in ``models.predict``. This exact
        path also powers ``POST /api/predict`` for Model A, so Model C receives the
        same predictor being evaluated rather than a second implementation. The
        model was fitted on 2023-24 and 2024-25; its live features are rebuilt only
        from records observable by ``as_of_date``.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD.
            as_of_date: ISO date for the prediction.
        """
        rejected = reject_unbound_request(
            "predict_win_probability",
            as_of_date=as_of_date,
            matchup_id=matchup_id,
        )
        if rejected:
            return rejected
        away, home, game_date = parse_matchup_id(matchup_id)
        if parse_date(as_of_date) >= game_date:
            return json.dumps(
                {
                    "status": "error",
                    "matchup_id": matchup_id,
                    "as_of_date": as_of_date,
                    "error": "as_of_date must be before the game date",
                },
                indent=2,
            )
        win = predict_win_model(home, away, as_of_date, game_date.isoformat())
        home_probability = win.get("home_win_prob")
        predicted_winner = None
        if win.get("status") == "ok" and home_probability is not None:
            predicted_winner = home if float(home_probability) >= 0.5 else away
        return json.dumps(
            {
                "status": win.get("status"),
                "model": win.get("model"),
                "matchup_id": matchup_id,
                "as_of_date": as_of_date,
                "home_team": home,
                "away_team": away,
                "home_win_prob": win.get("home_win_prob"),
                "away_win_prob": win.get("away_win_prob"),
                "predicted_winner": predicted_winner,
                "features": win.get("features"),
                "trained_on_seasons": win.get("trained_on_seasons"),
                "holdout_accuracy": win.get("holdout_accuracy"),
                "error": win.get("error"),
                "provenance": "same models.predict path as UI Model A",
            },
            indent=2,
        )

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
        rejected = reject_unbound_request(
            "predict_stat_line",
            as_of_date=as_of_date,
            matchup_id=matchup_id,
        )
        if rejected:
            return rejected
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

    @tool
    def predict_best_player(matchup_id: str, as_of_date: str) -> str:
        """Highest projected points among the gated rotation for this matchup.

        Uses predict_stat_line on players last seen on these two teams on or
        before as_of_date. Injured players are skipped. The candidate list is
        built from historical feature rows, never from the target box score.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD.
            as_of_date: ISO date. Same cutoff as every other tool.
        """
        rejected = reject_unbound_request(
            "predict_best_player",
            as_of_date=as_of_date,
            matchup_id=matchup_id,
        )
        if rejected:
            return rejected
        away, home, game_date = parse_matchup_id(matchup_id)
        if parse_date(as_of_date) >= game_date:
            return json.dumps(
                {
                    "status": "error",
                    "tool": "predict_best_player",
                    "error": "as_of_date must be before the game date",
                    "matchup_id": matchup_id,
                    "as_of_date": as_of_date,
                },
                indent=2,
            )
        rotation = source.observable_rotation(matchup_id, as_of_date)
        scored: list[dict] = []
        skipped: list[dict] = []
        for player in rotation:
            name = player["name"]
            if player_is_out(name, (away, home), parse_date(as_of_date)):
                skipped.append({"player": name, "reason": "listed Out as of cutoff"})
                continue
            if not stat_line_available():
                return _todo(
                    "predict_best_player",
                    "josh",
                    "models/stat_line.json is missing. Run "
                    "`python -m models.train_stat_line`.",
                    matchup_id=matchup_id,
                    as_of_date=as_of_date,
                )
            line = run_stat_line(source, name, matchup_id, as_of_date)
            if line.get("status") != "ok":
                skipped.append(
                    {
                        "player": name,
                        "reason": line.get("reason", line.get("status")),
                    }
                )
                continue
            scored.append(
                {
                    "player": name,
                    "team": player.get("team") or line.get("team"),
                    "projection": line.get("projection"),
                    "points_mae": line.get("points_mae"),
                }
            )
        scored.sort(
            key=lambda item: (
                -float((item.get("projection") or {}).get("points") or 0),
                item["player"],
            )
        )
        if not scored:
            return json.dumps(
                {
                    "status": "unavailable",
                    "tool": "predict_best_player",
                    "matchup_id": matchup_id,
                    "as_of_date": as_of_date,
                    "reason": (
                        "No gated rotation player produced a stat-line projection."
                    ),
                    "skipped": skipped,
                    "gated": True,
                },
                indent=2,
            )
        best = scored[0]
        return json.dumps(
            {
                "status": "ok",
                "tool": "predict_best_player",
                "matchup_id": matchup_id,
                "as_of_date": as_of_date,
                "best_player": best["player"],
                "team": best.get("team"),
                "projection": best.get("projection"),
                "points_mae": best.get("points_mae"),
                "ranked": scored[:5],
                "skipped": skipped,
                "uses": "predict_stat_line",
                "gated": True,
                "caveat": (
                    "Ranks the observable rotation by projected points. Same ridge "
                    "model and same as-of gate as predict_stat_line. Not part of "
                    "the required winner-path retrievals."
                ),
            },
            indent=2,
        )

    tools = [
        retrieve_matchup_context,
        retrieve_player_splits,
        retrieve_team_form,
        retrieve_injuries,
        predict_stat_line,
        predict_best_player,
    ]
    if include_model:
        tools.append(predict_win_probability)
    if without:
        unknown = set(without) - {t.name for t in tools}
        if unknown:
            raise ValueError(f"cannot withhold unknown tools: {sorted(unknown)}")
        tools = [t for t in tools if t.name not in without]
    return tools


def _stub_win_probability(
    source, home_abbr: str, away_abbr: str, as_of_date: str
) -> dict:
    """Net-rating + rest + injury heuristic. Placeholder until XGBoost lands.

    Reads ratings through the same source the agent uses, so it inherits the same
    date gating.
    """
    from agent.sources import (
        injuries_as_of,
        parse_date,
        season_end_year,
        team_form_as_of,
        team_ratings,
    )
    from agent.teams import normalize_abbr

    home_abbr, away_abbr = normalize_abbr(home_abbr), normalize_abbr(away_abbr)
    strength_basis = "mock fixture"

    if source.name == "real":
        as_of = parse_date(as_of_date)
        prior = season_end_year(as_of)
        home = team_ratings(home_abbr, prior - 1)
        away = team_ratings(away_abbr, prior - 1)
        if not home or not away:
            return {
                "model": "stub_net_rating_v0",
                "as_of_date": as_of_date,
                "home": home_abbr,
                "away": away_abbr,
                "home_win_prob": None,
                "away_win_prob": None,
                "error": "No prior-season ratings for one or both teams.",
            }

        # Prefer CURRENT-season rolling form over stale prior-season ratings.
        # avg_point_diff is a net-rating proxy that reflects who the team is now;
        # prior-season o_rtg/d_rtg is wrong by December. Fall back per team when
        # a team has no games yet (opening week).
        def net(abbr: str, prior_row: dict) -> float:
            form = team_form_as_of(abbr, as_of)
            if form and form["games_played"] >= 5:
                return form["avg_point_diff"]
            return prior_row["off_rating"] - prior_row["def_rating"]

        home_net, away_net = net(home_abbr, home), net(away_abbr, away)
        hf = team_form_as_of(home_abbr, as_of)
        strength_basis = (
            f"current form ({hf['record']}, {hf['avg_point_diff']:+.1f} pt diff)"
            if hf and hf["games_played"] >= 5
            else home.get("basis", "prior season")
        )
        rest_edge = 0.0  # real rest needs game logs; do not guess
    else:
        ctx = source.matchup_context(f"{away_abbr}-{home_abbr}-2026-01-15", as_of_date)
        home, away = ctx["home_team"], ctx["away_team"]
        home_net = home["off_rating"] - home["def_rating"]
        away_net = away["off_rating"] - away["def_rating"]
        rest_edge = 3.0 if ctx["rest"].get("away_back_to_back") else 0.0

    # Injury cost, weighted by who is actually out (advisor, 2026-07-21: a star
    # and a bench player used to count the same). 6.0 = net-rating points a full
    # replacement-level loss of one star is worth; deliberately a round number
    # until the model is fit.
    def injury_cost(abbr: str) -> tuple[float, list[dict]]:
        try:
            out = injuries_as_of(abbr, parse_date(as_of_date))
        except Exception:
            return 0.0, []
        weight = sum(i.get("importance") or 0.0 for i in out)
        return round(6.0 * weight, 2), out

    home_inj_cost, home_out = injury_cost(home_abbr)
    away_inj_cost, away_out = injury_cost(away_abbr)

    edge = (
        home_net
        - away_net
        + rest_edge
        + 2.5  # league-average home edge
        - home_inj_cost
        + away_inj_cost
    )
    home_win_prob = max(0.15, min(0.85, 0.5 + edge / 40.0))

    return {
        "model": "stub_net_rating_v2",
        "as_of_date": as_of_date,
        "home": home_abbr,
        "away": away_abbr,
        "home_win_prob": round(home_win_prob, 3),
        "away_win_prob": round(1.0 - home_win_prob, 3),
        "basis": strength_basis,
        "injury_impact": {
            "home_cost_net_rating": home_inj_cost,
            "away_cost_net_rating": away_inj_cost,
            "home_out": [
                {"player": i["player"], "tier": i.get("tier")} for i in home_out[:5]
            ],
            "away_out": [
                {"player": i["player"], "tier": i.get("tier")} for i in away_out[:5]
            ],
        },
        "warning": "Placeholder heuristic, not the XGBoost model. Injury weighting is "
        "a minutes/points proxy, not a fitted coefficient.",
    }


# Default tool set (mock) so `from agent.tools import TOOLS` still works.
TOOLS = build_tools(get_source("mock"))
