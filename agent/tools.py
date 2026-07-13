"""Tools the agent calls. Data comes from a Source (mock or real).

Signatures are the contract with the rest of the team (docs/tool-contracts.md)
and do not change when the underlying source does -- that is the whole point:
Kirtan/Patrick can swap what is behind retrieve_*, and the agent never notices.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.sources import get_source


def build_tools(source):
    """Bind a source (mock or real) into the tool set the agent gets."""

    @tool
    def retrieve_matchup_context(matchup_id: str, as_of_date: str) -> str:
        """Return team ratings, rest, injuries, and H2H for a matchup as of a date.

        Only records published on or before as_of_date are returned. Fields that
        cannot be computed from available data come back null with a reason --
        treat those as unknown, never as zero.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD, e.g. LAL-BOS-2026-01-15
            as_of_date: ISO date (YYYY-MM-DD). Nothing after this date is read.
        """
        return json.dumps(source.matchup_context(matchup_id, as_of_date), indent=2)

    @tool
    def retrieve_player_splits(player_name: str, back_to_back: bool = False) -> str:
        """Return a player's season averages, optionally their back-to-back split.

        Args:
            player_name: Full player name.
            back_to_back: If true, include the fatigue split when the source has one.
        """
        return json.dumps(source.player_splits(player_name, back_to_back), indent=2)

    @tool
    def predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str:
        """Stub win-probability model. Later this wraps Sarvvesh's XGBoost tool.

        Args:
            home_abbr: Home team abbreviation.
            away_abbr: Away team abbreviation.
            as_of_date: ISO date for the prediction.
        """
        return json.dumps(
            _stub_win_probability(source, home_abbr, away_abbr, as_of_date), indent=2
        )

    return [retrieve_matchup_context, retrieve_player_splits, predict_win_probability]


def _stub_win_probability(
    source, home_abbr: str, away_abbr: str, as_of_date: str
) -> dict:
    """Net-rating + rest heuristic. Placeholder until the XGBoost tool lands.

    Reads ratings through the same source the agent uses, so it inherits the
    same date gating.
    """
    from agent.sources import season_end_year, parse_date, team_ratings
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
        "note": "Placeholder until the XGBoost tool is wired.",
    }


# Default tool set (mock) so `from agent.tools import TOOLS` still works.
TOOLS = build_tools(get_source("mock"))
