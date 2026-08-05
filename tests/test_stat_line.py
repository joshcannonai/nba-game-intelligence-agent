"""The stat-line model: leakage guards on the data it reads and the dates it serves.

Two hazards, both of which have already bitten this project once in another form:

1. The features and the answer live in the same row. `player_stats_engineered.csv`
   carries `points` beside `rolling_pts_5`, exactly as the raw odds file carries
   `score_home` beside the line. The agent reads a stripped copy, and
   `test_inference_file_has_no_outcome_columns` is what keeps it stripped.

2. Trailing averages are only as-of if the game they trail is after as_of. Serving
   the row for a game that has already been played would hand the agent a window
   that includes games it is not allowed to see.
"""

from __future__ import annotations

import csv

import pytest

from agent.sources import (
    PLAYER_FEATURES_CSV,
    STAT_LINE_FEATURE_KEYS,
    CsvSource,
    get_source,
)

pytestmark = pytest.mark.skipif(
    not PLAYER_FEATURES_CSV.exists(),
    reason="player_features_2026.csv not built; run python -m models.stat_line_features",
)

# Anything that reveals how the game actually went. Kept as a literal list rather
# than imported from models/, so a change there cannot quietly relax this test.
FORBIDDEN = {
    "points",
    "total_rebounds",
    "assists",
    "outcome",
    "plus_minus",
    "game_score",
    "seconds_played",
    "minutes",
    "made_field_goals",
    "made_three_point_field_goals",
    "made_free_throws",
    "offensive_rebounds",
    "defensive_rebounds",
    "steals",
    "blocks",
    "turnovers",
}


def _header() -> list[str]:
    with PLAYER_FEATURES_CSV.open(encoding="utf-8") as f:
        return next(csv.reader(f))


def test_inference_file_has_no_outcome_columns():
    """The file the agent reads must not contain the thing it is predicting."""
    leaked = sorted(FORBIDDEN.intersection(_header()))
    assert not leaked, (
        f"{PLAYER_FEATURES_CSV.name} contains outcome columns {leaked}. "
        "Rebuild it with `python -m models.stat_line_features`."
    )


def test_inference_file_carries_every_feature():
    missing = [k for k in STAT_LINE_FEATURE_KEYS if k not in _header()]
    assert not missing, f"features absent from the inference file: {missing}"


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


def test_player_who_did_not_play_gets_a_reason_not_a_projection():
    """A missing box score is 'they did not play', never someone else's numbers."""
    result = CsvSource().player_features(
        "Nikola Jokić", "LAL-DEN-2025-12-02", "2025-12-01"
    )
    assert result["available"] is False
    assert "did not play" in result["reason"]


def test_served_row_is_the_game_being_predicted():
    result = CsvSource().player_features(
        "Nikola Jokić", "LAL-DEN-2025-12-03", "2025-12-02"
    )
    assert result["available"] is True
    assert result["game_date"] == "2025-12-03"
    assert set(result["features"]) == set(STAT_LINE_FEATURE_KEYS)
