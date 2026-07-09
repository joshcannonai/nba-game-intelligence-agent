"""Tool stubs the agent calls. Real date-gated retrieval and models plug in later."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

MOCK_DIR = Path(__file__).resolve().parents[1] / "data" / "mock"
DEFAULT_MATCHUP = MOCK_DIR / "matchup_lal_bos.json"


def _load_matchup(matchup_id: str | None = None) -> dict:
    path = DEFAULT_MATCHUP
    if matchup_id:
        candidate = MOCK_DIR / f"matchup_{matchup_id.lower().replace('-', '_')}.json"
        if candidate.exists():
            path = candidate
    with path.open() as f:
        return json.load(f)


@tool
def retrieve_matchup_context(matchup_id: str, as_of_date: str) -> str:
    """Return team ratings, rest, injuries, and H2H for a matchup as of a date.

    Args:
        matchup_id: e.g. LAL-BOS-2026-01-15
        as_of_date: ISO date (YYYY-MM-DD). Mock store ignores gating for now
            but the signature matches the real retrieve_*(as_of) contract.
    """
    data = _load_matchup(matchup_id)
    # Keep the as_of_date in the payload so the agent (and later harness) see it.
    payload = {
        "as_of_date": as_of_date,
        "matchup_id": data["matchup_id"],
        "game_date": data["game_date"],
        "home_team": data["home_team"],
        "away_team": data["away_team"],
        "rest": data["rest"],
        "injuries": [inj for inj in data["injuries"] if inj["published"] <= as_of_date],
        "h2h_last_5": [g for g in data["h2h_last_5"] if g["date"] <= as_of_date],
    }
    return json.dumps(payload, indent=2)


@tool
def retrieve_player_splits(player_name: str, back_to_back: bool = False) -> str:
    """Return season averages, optionally the player's back-to-back scoring split.

    Args:
        player_name: Full player name as it appears in the mock roster.
        back_to_back: If true, include b2b_pts_avg (fatigue context).
    """
    data = _load_matchup()
    for p in data["key_players"]:
        if p["name"].lower() == player_name.lower():
            out = {
                "name": p["name"],
                "team": p["team"],
                "pts_avg": p["pts_avg"],
                "reb_avg": p["reb_avg"],
                "ast_avg": p["ast_avg"],
            }
            if back_to_back:
                out["b2b_pts_avg"] = p["b2b_pts_avg"]
                out["note"] = (
                    "Player is on a back-to-back; prefer b2b split over season avg."
                )
            return json.dumps(out, indent=2)
    return json.dumps({"error": f"player not found: {player_name}"})


@tool
def predict_win_probability(home_abbr: str, away_abbr: str, as_of_date: str) -> str:
    """Stub win-probability model. Later this wraps Sarvvesh's XGBoost tool.

    Args:
        home_abbr: Home team abbreviation.
        away_abbr: Away team abbreviation.
        as_of_date: ISO date for the prediction (unused by stub, kept for contract).
    """
    data = _load_matchup()
    home = data["home_team"]
    away = data["away_team"]
    # Tiny heuristic so the stub is not a constant: net rating + rest.
    home_net = home["off_rating"] - home["def_rating"]
    away_net = away["off_rating"] - away["def_rating"]
    rest_edge = 3.0 if data["rest"]["away_back_to_back"] else 0.0
    edge = home_net - away_net + rest_edge
    # Map edge roughly into [0.35, 0.75]
    home_win_prob = max(0.35, min(0.75, 0.5 + edge / 40.0))
    return json.dumps(
        {
            "model": "stub_net_rating_v0",
            "as_of_date": as_of_date,
            "home": home_abbr,
            "away": away_abbr,
            "home_win_prob": round(home_win_prob, 3),
            "away_win_prob": round(1.0 - home_win_prob, 3),
            "note": "Placeholder until XGBoost tool is wired.",
        },
        indent=2,
    )


TOOLS = [retrieve_matchup_context, retrieve_player_splits, predict_win_probability]
