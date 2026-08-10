"""Contract tests for the model-only response used by the demo website."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.sources import get_source
from agent.tools import build_tools
from models import notebook_model_output as model_output
import models.predict as predict_module
from models.predict import predict as canonical_win_prediction
from ui.serve import app


MATCHUP_ID = "CHI-ORL-2025-12-01"
PREGAME_AS_OF = "2025-11-30"


def _assert_no_actual_results(value):
    if isinstance(value, dict):
        assert not any(key.startswith("actual_") for key in value)
        for child in value.values():
            _assert_no_actual_results(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_actual_results(child)


def test_predict_api_returns_predictions_without_completed_game_answers():
    response = TestClient(app).post(
        "/api/predict",
        json={"matchup_id": MATCHUP_ID, "as_of_date": PREGAME_AS_OF},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert 0.0 < payload["home_win_prob"] < 1.0
    assert payload["home_win_prob"] + payload["away_win_prob"] == pytest.approx(1.0)
    assert payload["player_stat_lines"]
    _assert_no_actual_results(payload)


def test_predict_api_uses_the_same_canonical_win_model_as_the_agent():
    response = TestClient(app).post(
        "/api/predict",
        json={"matchup_id": MATCHUP_ID, "as_of_date": PREGAME_AS_OF},
    )
    expected = canonical_win_prediction("ORL", "CHI", PREGAME_AS_OF)

    assert response.status_code == 200
    payload = response.json()
    assert payload["home_win_prob"] == expected["home_win_prob"]
    assert payload["away_win_prob"] == expected["away_win_prob"]
    assert payload["win_prediction"] == expected


def test_model_only_and_agent_report_the_same_missing_model_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing-model.json")
    predict_module.load_model.cache_clear()
    try:
        api_payload = model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)
        tools = {tool.name: tool for tool in build_tools(get_source("real"))}
        agent_payload = json.loads(
            tools["predict_win_probability"].invoke(
                {
                    "home_abbr": "ORL",
                    "away_abbr": "CHI",
                    "as_of_date": PREGAME_AS_OF,
                }
            )
        )

        assert api_payload["win_prediction"]["status"] == "awaiting_input"
        assert agent_payload == api_payload["win_prediction"]
    finally:
        predict_module.load_model.cache_clear()


def test_model_only_output_omits_players_known_out_before_tipoff():
    payload = model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)

    projected_names = {
        str(player.get("name", "")).casefold()
        for player in payload["player_stat_lines"]
    }
    assert "paolo banchero" not in projected_names


@pytest.mark.parametrize("as_of_date", ["2025-12-01", "2025-12-02"])
def test_predict_api_rejects_as_of_on_or_after_the_game(as_of_date):
    response = TestClient(app).post(
        "/api/predict",
        json={"matchup_id": MATCHUP_ID, "as_of_date": as_of_date},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "before the game" in payload["error"]


def test_target_game_player_rows_cannot_change_pregame_stat_lines(
    tmp_path, monkeypatch
):
    source = pd.read_csv(model_output.PLAYER_CSV)
    target = pd.to_datetime(source["game_date"]) == pd.Timestamp("2025-12-01")
    assert target.any()

    baseline = model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)[
        "player_stat_lines"
    ]
    without_target_results = source.loc[~target]
    csv_path = tmp_path / "without-target-player-results.csv"
    without_target_results.to_csv(csv_path, index=False)
    monkeypatch.setattr(model_output, "PLAYER_CSV", csv_path)

    assert model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)[
        "player_stat_lines"
    ] == baseline
