"""Gemma D/E must see the market. A/B/C must not. No GPU required."""

from __future__ import annotations

import json

from agent.sources import get_source
from agent.tools import build_tools
from eval.gemma_de import (
    BETTING_SKILL,
    PASSES,
    SCORE_COLUMNS,
    SYSTEM_D,
    SYSTEM_E,
    build_de_agent,
    pick_from_final,
    retrieve_betting_line_tool,
    sample_games,
    score_games,
    system_prompt_for_de,
)


def test_abc_still_cannot_see_the_market():
    names = {t.name for t in build_tools(get_source("mock"), include_model=True)}
    assert "retrieve_betting_line" not in names
    names_b = {t.name for t in build_tools(get_source("mock"), include_model=False)}
    assert "retrieve_betting_line" not in names_b


def test_thirty_sequential_passes_twenty_d_then_ten_e():
    assert len(PASSES) == 30
    assert [p["arm"] for p in PASSES].count("D") == 20
    assert [p["arm"] for p in PASSES].count("E") == 10
    assert PASSES[0]["arm"] == "D"
    assert PASSES[20]["arm"] == "E"


def test_d_prompt_is_winners_e_prompt_is_money():
    d = SYSTEM_D.lower()
    e = SYSTEM_E.lower()
    assert "pick winners" in d or "maximize accuracy" in d
    assert "retrieve_betting_line" in d
    assert "make money" in e
    assert "pick=none" in e
    assert "you have no betting-line tool" not in d
    assert "you have no betting-line tool" not in e


def test_de_prompt_includes_inline_betting_skill_not_abc_ban():
    names = [t.name for t in build_tools(get_source("mock"), include_model=True)]
    names.append("retrieve_betting_line")
    prompt = system_prompt_for_de("winner", names, "")
    assert "retrieve_betting_line" in prompt
    assert BETTING_SKILL.strip() in prompt
    assert "You have no betting-line tool" not in prompt


def test_pick_from_final_supports_none_and_prob_fallback():
    assert pick_from_final({"pick": "HOME", "home_win_prob": 0.4}) is True
    assert pick_from_final({"pick": "AWAY", "home_win_prob": 0.9}) is False
    assert pick_from_final({"pick": "NONE"}) is None
    assert pick_from_final({"home_win_prob": 0.51}) is True
    assert pick_from_final({"home_win_prob": 0.49}) is False
    assert (
        pick_from_final({"pick": "GSW", "home_win_prob": 0.9}, home="LAL", away="GSW")
        is False
    )
    assert (
        pick_from_final({"pick": "LAL", "home_win_prob": 0.1}, home="LAL", away="GSW")
        is True
    )


def test_score_games_counts_no_bet_as_zero_pnl_not_a_miss():
    rows = [
        {
            "pick_home": True,
            "correct": 1,
            "pnl": 80.0,
            "vegas_correct": 1,
            "error": None,
        },
        {
            "pick_home": None,
            "correct": None,
            "pnl": 0.0,
            "vegas_correct": 0,
            "error": None,
        },
        {
            "pick_home": False,
            "correct": 0,
            "pnl": -100.0,
            "vegas_correct": 1,
            "error": None,
        },
    ]
    out = score_games(rows)
    assert out["n_games"] == 3
    assert out["n_bets"] == 2
    assert out["correct"] == 1
    assert out["accuracy"] == 0.5
    assert out["net_pnl"] == -20.0


def test_sample_is_gated_2026_with_odds_and_results():
    games = sample_games(8)
    assert len(games) == 8
    assert len({g["game_id"] for g in games}) == 8
    for g in games:
        assert g["cutoff"] < g["game_date"]
        assert 0 < g["p_market"] < 1
        assert isinstance(g["home_won"], bool)


def test_betting_line_tool_has_no_score_columns():
    source = get_source("real")
    tool = retrieve_betting_line_tool(source, "2024-12-22", "DET-LAL-2024-12-23")
    payload = json.loads(
        tool.invoke({"matchup_id": "DET-LAL-2024-12-23", "as_of_date": "2024-12-22"})
    )
    assert payload["status"] == "ok", payload
    assert not (SCORE_COLUMNS & payload.keys())
    rejected = json.loads(
        tool.invoke({"matchup_id": "LAL-BOS-2024-12-25", "as_of_date": "2024-12-22"})
    )
    assert rejected["status"] == "error"
    assert rejected["data_source_read"] is False


def test_de_agent_constructs_gemma4_with_betting_line(monkeypatch):
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

    agent = build_de_agent(
        get_source("mock"),
        objective="winner",
        prompt_addendum="\n- extra rule",
        required_as_of_date="2024-12-24",
        required_matchup_id="LAL-BOS-2024-12-25",
    )
    assert agent == "fake-agent"
    assert captured["kwargs"]["model"] == "gemma4"
    assert captured["kwargs"]["temperature"] == 0
    assert "retrieve_betting_line" in captured["tools"]
    assert "predict_win_probability" in captured["tools"]
    assert "extra rule" in captured["prompt"]
    assert "You have no betting-line tool" not in captured["prompt"]
