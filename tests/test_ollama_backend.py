"""B and C must actually construct the local Gemma chat model.

A previous bug had Model B reading skill files without ever calling Ollama.
This test patches ChatOllama and create_agent so it can fail without loading
weights or using RAM.
"""

from __future__ import annotations

from agent.run import build_agent
from agent.sources import get_source


def test_ollama_backend_constructs_chatollama(monkeypatch):
    captured: dict = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    def fake_create_agent(model, tools, system_prompt=None):
        captured["model"] = model
        captured["tools"] = [t.name for t in tools]
        captured["prompt"] = system_prompt
        return "fake-agent"

    monkeypatch.setattr("langchain_ollama.ChatOllama", FakeChatOllama)
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    source = get_source("mock")
    agent = build_agent(source, model_backend="ollama", include_model=False)

    assert agent == "fake-agent"
    assert captured["kwargs"]["model"] == "gemma4"
    assert captured["kwargs"]["temperature"] == 0
    assert captured["kwargs"]["reasoning"] is False
    assert "predict_win_probability" not in captured["tools"]
    assert "retrieve_betting_line" not in captured["tools"]
    assert (
        "--- SKILL FILE: skills/predict_win_probability.md ---"
        not in captured["prompt"]
    )


def test_ollama_model_c_gets_the_predictor_and_may_disagree(monkeypatch):
    captured: dict = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    def fake_create_agent(model, tools, system_prompt=None):
        captured["tools"] = [t.name for t in tools]
        captured["prompt"] = system_prompt
        return "fake-agent"

    monkeypatch.setattr("langchain_ollama.ChatOllama", FakeChatOllama)
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    build_agent(get_source("mock"), model_backend="ollama", include_model=True)

    assert captured["kwargs"]["model"] == "gemma4"
    assert "predict_win_probability" in captured["tools"]
    assert "retrieve_betting_line" not in captured["tools"]
    prompt = " ".join(captured["prompt"].lower().split())
    assert "may agree or disagree" in prompt
    assert "--- skill file: skills/predict_win_probability.md ---" in prompt
