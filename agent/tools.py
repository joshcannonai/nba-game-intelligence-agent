"""The tools the agent can call. This file IS the contract with the rest of the team.

Every function the agent needs in order to produce the report we described on 7/07
(who wins · best player · a narrative · statistics · the betting line) exists here
NOW, with a stable name and a stable signature.

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

from agent.sources import get_source


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


def build_tools(source):
    """Bind a data source into the tool set the agent gets."""

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
        return _todo(
            "retrieve_schedule",
            "Patrick (already pulled -- needs committing)",
            "A forward-looking slate: date, home, away, tip-off time. Patrick's "
            "data/pull_games.py writes season_schedule_2026.csv and it has still not "
            "been committed -- data/raw stopped being gitignored on 7/21, so there is "
            "no longer anything in the way. data/samples/game_logs_*.csv carries "
            "date/home/away for finished seasons and could back this tool today; what "
            "it lacks is tip-off times and any game that has not been played yet.",
            as_of_date=as_of_date,
            days_ahead=days_ahead,
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

    @tool
    def retrieve_news(team_abbr: str, as_of_date: str, limit: int = 5) -> str:
        """Beat-reporter news and narrative for a team, published on or before a date.

        The qualitative half of the report -- the "story" of the game. This is the
        ESPN / RotoWire source we named on 7/07 and nobody has started.

        Args:
            team_abbr: Team abbreviation.
            as_of_date: ISO date. Nothing published after this may be returned.
            limit: Max items.
        """
        return _todo(
            "retrieve_news",
            "Josh (scope-cut candidate)",
            "Scraped articles/notes each carrying a PUBLICATION TIMESTAMP, so they can "
            "be filtered to as_of_date. Not started, and proposed for the Week-4 scope "
            "cut: highest effort, lowest measurable contribution of the ten.",
            team_abbr=team_abbr,
            as_of_date=as_of_date,
        )

    @tool
    def retrieve_betting_line(matchup_id: str, as_of_date: str) -> str:
        """The market's closing price on this game: spread, total, moneyline.

        CONTEXT ONLY -- do NOT let this drive the win probability. The advisor's
        call on 2026-07-21: the line is an evaluation baseline, not a model input,
        or the system just reads the answer off the market instead of predicting.

        Reads data/samples/odds_only.csv, which is built from an allowlist of safe
        columns (scripts/odds_only.py), so this tool structurally cannot leak the
        result even if it wanted to.

        It is the CLOSING line, so it is not knowable until tip-off. Report it and
        compare against it; do not treat it as something we knew on as_of_date, and
        do not let it stand in for a prediction of our own.

        spread is an unsigned magnitude; favorite, spread_home and spread_away say
        who is laying the points. A query dated after tip-off is refused outright.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD
            as_of_date: ISO date. Must be on or before tip-off.
        """
        payload = source.betting_line(matchup_id, as_of_date)
        return json.dumps(
            {
                "matchup_id": matchup_id,
                "as_of_date": as_of_date,
                "usage": "Evaluation baseline and narrative context only. Do not "
                "use this to set the win probability.",
                **payload,
            }
        )

    # ---------------------------------------------------- MODELS (Sarvesh) TODO

    @tool
    def predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str:
        """Probability the home team wins. PLACEHOLDER: form, rest and injuries.

        Sarvesh's XGBoost replaces the body of this. Known weakness of the placeholder:
        the injury term weights players by their PRIOR season's minutes and treats
        every listed player as fully out, so it over-penalises. Measured across all
        1,322 games of 2025-26 it costs accuracy rather than adding it -- 63.4% with
        injuries against 66.3% without.

        Args:
            home_abbr: Home team abbreviation.
            away_abbr: Away team abbreviation.
            as_of_date: ISO date for the prediction.
        """
        return json.dumps(
            _stub_win_probability(source, home_abbr, away_abbr, as_of_date), indent=2
        )

    @tool
    def predict_stat_line(player_name: str, matchup_id: str, as_of_date: str) -> str:
        """Projected points / rebounds / assists for one player in this game.

        The "statistics" half of the report we pitched.

        Args:
            player_name: Full player name.
            matchup_id: AWAY-HOME-YYYY-MM-DD
            as_of_date: ISO date.
        """
        return _todo(
            "predict_stat_line",
            "Sarvesh (linear regression)",
            "The stat-line regression from the PDP. Not started.",
            player_name=player_name,
            matchup_id=matchup_id,
            as_of_date=as_of_date,
        )

    @tool
    def predict_best_player(matchup_id: str, as_of_date: str) -> str:
        """Who is likely to be the best player in this game.

        Explicitly part of the output we described to Sadovnik on 7/07
        ("who wins, best player, a narrative, statistics, a betting line").

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD
            as_of_date: ISO date.
        """
        return _todo(
            "predict_best_player",
            "Sarvesh (models)",
            "Ranks likely top performers. Depends on predict_stat_line. Not started.",
            matchup_id=matchup_id,
            as_of_date=as_of_date,
        )

    return [
        retrieve_matchup_context,
        retrieve_player_splits,
        retrieve_schedule,
        retrieve_team_form,
        retrieve_injuries,
        retrieve_news,
        retrieve_betting_line,
        predict_win_probability,
        predict_stat_line,
        predict_best_player,
    ]


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
