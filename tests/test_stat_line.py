"""The stat-line model: leakage guards on the data it reads and the dates it serves.

Two hazards, both of which have already bitten this project once in another form:

1. Player outcomes are useful for recomputing form, but only rows dated on or
   before `as_of` may be read. Snapshot tests physically remove later rows, while
   these tests verify the query-time boundary.

2. Trailing averages are only as-of if the game they trail is after as_of. Serving
   the row for a game that has already been played would hand the agent a window
   that includes games it is not allowed to see.
"""

from __future__ import annotations

from datetime import date
import json

import pytest

from agent.sources import (
    PLAYER_STATS_CSV,
    STAT_LINE_FEATURE_KEYS,
    CsvSource,
    get_source,
    parse_date,
)
from agent.tools import build_tools


def test_stat_line_tool_suppresses_projection_for_player_known_out():
    tools = {tool.name: tool for tool in build_tools(get_source("real"))}

    payload = tools["predict_stat_line"].invoke(
        {
            "player_name": "Paolo Banchero",
            "matchup_id": "CHI-ORL-2025-12-01",
            "as_of_date": "2025-11-30",
        }
    )

    result = json.loads(payload)
    assert result["status"] == "unavailable"
    assert result["player"] == "Paolo Banchero"
    assert "listed Out" in result["reason"]
    assert "projection" not in result


pytestmark = pytest.mark.skipif(
    not PLAYER_STATS_CSV.exists(),
    reason="player_stats_engineered.csv is missing",
)

def test_feature_lists_agree_between_data_and_model_layers():
    """Feature order is positional in stat_line.json. Divergence would mis-score."""
    from models.stat_line_features import FEATURE_NAMES

    assert tuple(FEATURE_NAMES) == tuple(STAT_LINE_FEATURE_KEYS)


def test_as_of_on_or_after_tip_off_is_refused():
    source = CsvSource()
    with pytest.raises(ValueError, match="not before tip-off"):
        source.player_features("Nikola Jokić", "LAL-DEN-2025-12-03", "2025-12-03")


def test_mock_source_refuses_after_tip_off_identically():
    """Mock and real must reject the same query, or tests pass for the wrong reason."""
    with pytest.raises(ValueError, match="not before tip-off"):
        get_source("mock").player_features(
            "Nikola Jokić", "LAL-DEN-2025-12-03", "2025-12-03"
        )


def test_player_features_use_a_snapshot_on_or_before_as_of():
    """Projection availability must not depend on a future box-score row."""
    result = CsvSource().player_features(
        "Nikola Jokić", "LAL-DEN-2025-12-02", "2025-12-01"
    )
    assert result["available"] is True
    assert result["game_date"] == "2025-12-02"
    assert parse_date(result["feature_snapshot_date"]) <= date(2025, 12, 1)


def test_early_as_of_cannot_use_an_intervening_game():
    result = CsvSource().player_features(
        "Nikola Jokić", "LAL-DEN-2025-12-05", "2025-12-02"
    )
    assert result["available"] is True
    assert parse_date(result["feature_snapshot_date"]) <= date(2025, 12, 2)


def test_no_snapshot_does_not_reveal_whether_player_appears_later():
    source = CsvSource()
    future_player = source.player_features(
        "Precious Achiuwa", "HOU-OKC-2025-10-21", "2025-10-20"
    )
    absent_player = source.player_features(
        "Definitely Not A Player", "HOU-OKC-2025-10-21", "2025-10-20"
    )
    assert future_player == absent_player


def test_served_row_is_the_game_being_predicted():
    result = CsvSource().player_features(
        "Nikola Jokić", "LAL-DEN-2025-12-03", "2025-12-02"
    )
    assert result["available"] is True
    assert result["game_date"] == "2025-12-03"
    assert parse_date(result["feature_snapshot_date"]) <= date(2025, 12, 2)
    assert set(result["features"]) == set(STAT_LINE_FEATURE_KEYS)
