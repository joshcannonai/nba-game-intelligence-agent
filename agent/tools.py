"""The tools the agent can call. This file IS the contract with the rest of the team.

Every function the agent needs in order to produce the report we described on 7/07
(who wins · best player · a narrative · statistics · the betting line) exists here
NOW, with a stable name and a stable signature.

Some of them work. Most of them are placeholders that return:

    {"status": "not_implemented", "owner": "...", "needs": "..."}

That is deliberate. A placeholder is not a stub that lies -- it tells the agent, in
plain terms, that the data does not exist yet and who is building it. The agent then
reports the gap in its output instead of inventing an answer. So running the agent
today prints an honest status board of the whole project.

To implement one: keep the name and the arguments, replace the body. The agent never
notices -- that is the entire point of putting the interface in first.

    data layer   (Patrick + Kirtan) -> everything named retrieve_*
    models       (Sarvvesh)         -> everything named predict_*
    agent        (Josh)             -> the loop that calls them
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.sources import get_source


def _todo(tool_name: str, owner: str, needs: str, **ctx) -> str:
    """The honest placeholder. Never fabricates, always names an owner."""
    return json.dumps(
        {
            "status": "not_implemented",
            "tool": tool_name,
            "owner": owner,
            "needs": needs,
            **ctx,
            "note": (
                "This tool is not built yet. Report it as unavailable in your output. "
                "Do NOT invent a value and do NOT treat it as zero."
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
            "A game-by-game schedule table: date, home, away, tip-off time. Patrick's "
            "data/pull_games.py writes season_schedule_2026.csv, but it lands in "
            "data/raw/ which is gitignored, so it never reaches the repo. Commit the "
            "derived table (or un-ignore a data/curated/ path) and this tool works.",
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
        return _todo(
            "retrieve_team_form",
            "Josh (not in the PDP -- found while building)",
            "A rolling, as-of team rating computed from games played BEFORE as_of_date "
            "(rolling net rating or Elo). The PDP never specced this: we assumed the "
            "season CSVs would serve, and they cannot without leaking. Needs the "
            "game-by-game table first.",
            team_abbr=team_abbr,
            as_of_date=as_of_date,
            last_n=last_n,
        )

    @tool
    def retrieve_injuries(team_abbr: str, as_of_date: str) -> str:
        """Who was KNOWN to be out, on the morning of the game.

        Works today by replaying the injury transaction log and stopping at as_of_date.
        Two known limits: the log ends 2025-01-12, and it gives no measure of how much
        a player matters -- a 10th man and a franchise player weigh the same.

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
        """The market's price on this game: spread, moneyline, total.

        This is our evaluation baseline (Sadovnik, 7/07 -- beating the line is a better
        signal than beating the result, because games have upsets). The line is also
        what the agent's prediction gets compared against.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD
            as_of_date: ISO date. Use the line as it stood on that date.
        """
        return _todo(
            "retrieve_betting_line",
            "Josh (have the data, wiring it up)",
            "Historical odds per game (spread, moneyline, total). Covered: 24,441 games "
            "2008-2026, including all 1,322 games of 2025-26 and its 85 playoff games. "
            "LEAKAGE NOTE: the file carries score_away/score_home in the same row as the "
            "line -- this tool must return the line columns ONLY.",
            matchup_id=matchup_id,
            as_of_date=as_of_date,
        )

    # ---------------------------------------------------- MODELS (Sarvvesh) TODO

    @tool
    def predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str:
        """Probability the home team wins. PLACEHOLDER: a net-rating + rest heuristic.

        Sarvvesh's XGBoost replaces the body of this. Known weakness of the placeholder:
        it ignores the injury list entirely, so the prediction does not move when the
        injury list does.

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
            "Sarvvesh (linear regression)",
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
            "Sarvvesh (models)",
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
    """Net-rating + rest heuristic. Placeholder until the XGBoost tool lands.

    Reads ratings through the same source the agent uses, so it inherits the same
    date gating.
    """
    from agent.sources import parse_date, season_end_year, team_ratings
    from agent.teams import normalize_abbr

    home_abbr, away_abbr = normalize_abbr(home_abbr), normalize_abbr(away_abbr)

    if source.name == "real":
        prior = season_end_year(parse_date(as_of_date))
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
        rest_edge = 0.0  # real rest needs game logs; do not guess
    else:
        ctx = source.matchup_context(f"{away_abbr}-{home_abbr}-2026-01-15", as_of_date)
        home, away = ctx["home_team"], ctx["away_team"]
        rest_edge = 3.0 if ctx["rest"].get("away_back_to_back") else 0.0

    home_net = home["off_rating"] - home["def_rating"]
    away_net = away["off_rating"] - away["def_rating"]
    edge = home_net - away_net + rest_edge + 2.5  # 2.5 = league-average home edge
    home_win_prob = max(0.15, min(0.85, 0.5 + edge / 40.0))

    return {
        "model": "stub_net_rating_v0",
        "as_of_date": as_of_date,
        "home": home_abbr,
        "away": away_abbr,
        "home_win_prob": round(home_win_prob, 3),
        "away_win_prob": round(1.0 - home_win_prob, 3),
        "basis": home.get("basis", "mock fixture"),
        "warning": "Placeholder. Ignores injuries entirely -- see predict_win_probability.",
    }


# Default tool set (mock) so `from agent.tools import TOOLS` still works.
TOOLS = build_tools(get_source("mock"))
