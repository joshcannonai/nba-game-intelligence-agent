"""Contract tests for the model-only response used by the demo website."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from models import notebook_model_output as model_output
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
    assert payload["trained_on_seasons"] == [2024, 2025]
    _assert_no_actual_results(payload)


def test_clippers_abbreviation_matches_the_exported_team_key():
    payload = model_output.predict_model_only("GSW-LAC-2026-04-15", "2026-04-05")
    assert payload["status"] == "ok"
    assert payload["home"] == "LOS_ANGELES_CLIPPERS"


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


def test_target_game_final_score_cannot_change_the_pregame_prediction(
    tmp_path, monkeypatch
):
    source = pd.read_csv(model_output.TEAM_CSV)
    target_date = pd.Timestamp("2025-12-01")
    dates = pd.to_datetime(source["game_date"])
    home = (dates == target_date) & (source["team"] == "ORLANDO_MAGIC")
    away = (dates == target_date) & (source["team"] == "CHICAGO_BULLS")
    assert home.sum() == 1
    assert away.sum() == 1

    def prediction_with_final_score(
        filename: str,
        *,
        home_for: int,
        home_against: int,
        away_for: int,
        away_against: int,
    ) -> float:
        changed = source.copy()
        changed.loc[home, ["points_scored", "points_allowed"]] = [
            home_for,
            home_against,
        ]
        changed.loc[away, ["points_scored", "points_allowed"]] = [
            away_for,
            away_against,
        ]
        csv_path = tmp_path / filename
        changed.to_csv(csv_path, index=False)
        monkeypatch.setattr(model_output, "TEAM_CSV", csv_path)
        return model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)[
            "home_win_prob"
        ]

    home_loss = prediction_with_final_score(
        "home-loss.csv",
        home_for=0,
        home_against=250,
        away_for=250,
        away_against=0,
    )
    home_win = prediction_with_final_score(
        "home-win.csv",
        home_for=250,
        home_against=0,
        away_for=0,
        away_against=250,
    )

    assert home_loss == pytest.approx(home_win, abs=1e-12)


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

    assert (
        model_output.predict_model_only(MATCHUP_ID, PREGAME_AS_OF)["player_stat_lines"]
        == baseline
    )


def test_early_as_of_team_prediction_ignores_intervening_results(tmp_path, monkeypatch):
    source = pd.read_csv(model_output.TEAM_CSV)
    early_as_of = "2025-11-20"
    baseline = model_output.predict_model_only(MATCHUP_ID, early_as_of)["home_win_prob"]
    dates = pd.to_datetime(source["game_date"])
    future = dates > pd.Timestamp(early_as_of)
    changed = source.copy()
    changed.loc[future, ["points_scored", "points_allowed", "won"]] = [250, 0, 1]
    csv_path = tmp_path / "mutated-intervening-results.csv"
    changed.to_csv(csv_path, index=False)
    monkeypatch.setattr(model_output, "TEAM_CSV", csv_path)

    assert model_output.predict_model_only(MATCHUP_ID, early_as_of)[
        "home_win_prob"
    ] == pytest.approx(baseline, abs=1e-12)
