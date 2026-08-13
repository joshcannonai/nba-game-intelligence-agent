"""Professor-facing proof emitted by the live /api/run stream."""

from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient

import ui.serve as serve


class FakeAgent:
    def stream(self, _input, stream_mode):
        assert stream_mode == "updates"
        yield {
            "model": {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content="",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "retrieve_team_form",
                                "args": {
                                    "team_abbr": "ORL",
                                    "as_of_date": "2025-11-30",
                                },
                            }
                        ],
                        usage_metadata={"input_tokens": 120, "output_tokens": 15},
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    SimpleNamespace(
                        type="tool",
                        name="retrieve_team_form",
                        tool_call_id="call-1",
                        content='{"record":"8-2","as_of":"2025-11-30"}',
                        tool_calls=[],
                    )
                ]
            }
        }
        yield {
            "model": {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content='{"home_win_prob":0.84}',
                        tool_calls=[],
                        usage_metadata={"input_tokens": 190, "output_tokens": 24},
                    )
                ]
            }
        }


class FakeListResultAgent:
    def stream(self, _input, stream_mode):
        assert stream_mode == "updates"
        yield {
            "model": {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content="",
                        tool_calls=[
                            {
                                "id": "call-list",
                                "name": "retrieve_injuries",
                                "args": {
                                    "team_abbr": "ORL",
                                    "as_of_date": "2025-11-30",
                                },
                            }
                        ],
                        usage_metadata={},
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    SimpleNamespace(
                        type="tool",
                        name="retrieve_injuries",
                        tool_call_id="call-list",
                        content="[]",
                        tool_calls=[],
                    )
                ]
            }
        }
        yield {
            "model": {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content='{"home_win_prob":0.5}',
                        tool_calls=[],
                        usage_metadata={},
                    )
                ]
            }
        }


def test_run_stream_exposes_exact_inspectable_context(monkeypatch):
    monkeypatch.setattr(serve, "build_agent", lambda *_args, **_kwargs: FakeAgent())
    monkeypatch.setattr(
        serve, "runtime_system_prompt", lambda *_args, **_kwargs: "SYSTEM RULES"
    )

    response = TestClient(serve.app).post(
        "/api/run",
        json={
            "matchup_id": "CHI-ORL-2025-12-01",
            "as_of_date": "2025-11-30",
            "include_model": True,
            "model_backend": "ollama",
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "event: context_start" in body
    assert "event: context_message" in body
    assert '"role": "system", "content": "SYSTEM RULES"' in body
    assert '"role": "user"' in body
    assert '"role": "assistant"' in body
    assert '"role": "tool"' in body
    assert '"tool_call_id": "call-1"' in body
    assert '"input_tokens": 310' in body
    assert '"output_tokens": 39' in body
    assert "event: gate_receipt" in body
    assert '"tool": "retrieve_team_form"' in body
    assert '"requested_cutoff": "2025-11-30"' in body
    assert '"latest_historical_date": "2025-11-30"' in body
    assert '"post_cutoff_records": 0' in body
    assert '"status": "passed"' in body
    assert "event: tool_result" in body
    assert '"status": "ok"' in body


def test_gate_receipt_fails_when_the_tool_omits_the_cutoff():
    """The old receipt treated a missing as_of_date as a pass via an OR."""
    receipt = serve._gate_receipt(
        SimpleNamespace(
            name="retrieve_team_form",
            content='{"record":"8-2","as_of":"2025-11-30"}',
        ),
        "2025-11-30",
        "real",
        {"args": {}},
    )
    assert receipt["status"] == "failed"
    assert receipt["tool_cutoff"] is None


def test_gate_receipt_fails_when_the_tool_cutoff_differs():
    receipt = serve._gate_receipt(
        SimpleNamespace(
            name="retrieve_team_form",
            content='{"record":"8-2","as_of":"2025-12-01"}',
        ),
        "2025-11-30",
        "real",
        {"args": {"as_of_date": "2025-12-01"}},
    )
    assert receipt["status"] == "failed"
    assert receipt["tool_cutoff"] == "2025-12-01"


def test_run_stream_accepts_a_valid_list_tool_payload(monkeypatch):
    monkeypatch.setattr(
        serve, "build_agent", lambda *_args, **_kwargs: FakeListResultAgent()
    )
    monkeypatch.setattr(
        serve, "runtime_system_prompt", lambda *_args, **_kwargs: "SYSTEM RULES"
    )
    response = TestClient(serve.app).post(
        "/api/run",
        json={
            "matchup_id": "CHI-ORL-2025-12-01",
            "as_of_date": "2025-11-30",
            "include_model": False,
            "model_backend": "none",
        },
    )
    assert response.status_code == 200
    assert "event: error" not in response.text
    assert '"name": "retrieve_injuries", "status": "ok"' in response.text


def test_run_rejects_a_non_pregame_cutoff_before_building_the_agent(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("agent must not be built for an invalid cutoff")

    monkeypatch.setattr(serve, "build_agent", forbidden)
    response = TestClient(serve.app).post(
        "/api/run",
        json={
            "matchup_id": "CHI-ORL-2025-12-01",
            "as_of_date": "2025-12-01",
            "include_model": True,
            "model_backend": "ollama",
        },
    )

    assert response.status_code == 422
    assert "before the game" in response.json()["detail"]


def test_runtime_fingerprint_changes_when_a_dataset_changes(tmp_path, monkeypatch):
    from agent import sources

    dataset = tmp_path / "games.csv"
    dataset.write_text("game_id\na\n", encoding="utf-8")
    monkeypatch.setattr(sources, "TEAM_SUMMARY_CSV", dataset)
    monkeypatch.setattr(sources, "PLAYER_PER_GAME_CSV", Path("/missing/player.csv"))
    monkeypatch.setattr(sources, "ODDS_CSV", Path("/missing/odds.csv"))
    monkeypatch.setattr(sources, "INJURY_CSVS", ())
    monkeypatch.setattr(sources, "SAMPLE_DIR", tmp_path / "no-logs")
    first = serve.runtime_fingerprint("none")["sha256"]
    dataset.write_text("game_id\na\nb\n", encoding="utf-8")
    second = serve.runtime_fingerprint("none")["sha256"]
    assert first != second
