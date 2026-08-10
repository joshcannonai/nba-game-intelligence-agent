"""Professor-facing proof emitted by the live /api/run stream."""

from types import SimpleNamespace

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
                                "args": {"team_abbr": "ORL", "as_of_date": "2025-11-30"},
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


def test_run_stream_exposes_exact_inspectable_context(monkeypatch):
    monkeypatch.setattr(serve, "build_agent", lambda *_args, **_kwargs: FakeAgent())
    monkeypatch.setattr(serve, "runtime_system_prompt", lambda *_args, **_kwargs: "SYSTEM RULES")

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
