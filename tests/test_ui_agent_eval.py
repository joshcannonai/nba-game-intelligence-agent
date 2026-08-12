from __future__ import annotations

import csv
import json
from datetime import date, timedelta

import pytest

from eval.ui_agent_eval import (
    FULL_SEASON_CONTRACT,
    REQUIRED_CALLS,
    ROOT,
    SAMPLE_CONTRACT,
    assert_real_ui_result,
    cutoff_for_game,
    game_truth,
    read_checkpoint,
)


def test_full_season_uses_every_canonical_game_once():
    path = ROOT / "data" / "samples" / "game_logs_2026.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        ids = [row["game_id"] for row in csv.DictReader(fh)]

    assert len(ids) == 1322
    assert len(set(ids)) == 1322
    assert set(game_truth(ids)) == set(ids)


def test_full_season_cutoff_is_previous_calendar_day():
    truth = {"game_date": "2025-10-21"}
    cutoff = cutoff_for_game(truth, fixed_cutoff=None)

    assert cutoff == "2025-10-20"
    assert date.fromisoformat(cutoff) == date.fromisoformat(
        truth["game_date"]
    ) - timedelta(days=1)


def test_sample_mode_keeps_the_explicit_shared_cutoff():
    truth = {"game_date": "2026-04-07"}
    assert cutoff_for_game(truth, fixed_cutoff="2026-04-05") == "2026-04-05"


def test_sample_and_full_season_have_different_contracts():
    assert SAMPLE_CONTRACT != FULL_SEASON_CONTRACT


def test_b_and_c_require_the_same_retrievals_and_c_adds_only_model_a():
    assert REQUIRED_CALLS["C"] == {
        **REQUIRED_CALLS["B"],
        "predict_win_probability": 1,
    }


def test_rows_from_a_different_runtime_contract_are_not_resumed(tmp_path):
    contracts = {
        "A": {"runtime_contract_sha256": "a", "system_prompt_sha256": "not_applicable"},
        "B": {"runtime_contract_sha256": "b", "system_prompt_sha256": "prompt-b"},
        "C": {"runtime_contract_sha256": "c", "system_prompt_sha256": "prompt-c"},
    }
    path = tmp_path / "checkpoint.jsonl"
    rows = [
        {"arm": "A", "game_id": "g1"},
        {"arm": "A", "game_id": "g2", "eval_contract": FULL_SEASON_CONTRACT},
        {
            "arm": "A",
            "game_id": "g3",
            "eval_contract": FULL_SEASON_CONTRACT,
            "runtime_contract_sha256": "a",
            "system_prompt_sha256": "not_applicable",
        },
        {
            "arm": "B",
            "game_id": "g1",
            "eval_contract": FULL_SEASON_CONTRACT,
            "runtime_contract_sha256": "old-b",
            "system_prompt_sha256": "prompt-b",
        },
        {
            "arm": "B",
            "game_id": "g2",
            "eval_contract": FULL_SEASON_CONTRACT,
            "runtime_contract_sha256": "b",
            "system_prompt_sha256": "old-prompt",
        },
        {
            "arm": "B",
            "game_id": "g3",
            "eval_contract": FULL_SEASON_CONTRACT,
            "runtime_contract_sha256": "b",
            "system_prompt_sha256": "prompt-b",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    resumed = read_checkpoint(path, FULL_SEASON_CONTRACT, contracts)

    assert ("A", "g1") not in resumed
    assert ("A", "g2") not in resumed
    assert ("A", "g3") in resumed
    assert ("B", "g1") not in resumed
    assert ("B", "g2") not in resumed
    assert ("B", "g3") in resumed


def _valid_agent_result() -> dict:
    game_id, cutoff = "HOU-OKC-2025-10-21", "2025-10-20"
    calls = [
        {
            "name": "retrieve_matchup_context",
            "args": {"matchup_id": game_id, "as_of_date": cutoff},
        },
        {
            "name": "retrieve_team_form",
            "args": {"team_abbr": "HOU", "as_of_date": cutoff, "last_n": 10},
        },
        {
            "name": "retrieve_team_form",
            "args": {"team_abbr": "OKC", "as_of_date": cutoff, "last_n": 10},
        },
        {
            "name": "retrieve_injuries",
            "args": {"team_abbr": "HOU", "as_of_date": cutoff},
        },
        {
            "name": "retrieve_injuries",
            "args": {"team_abbr": "OKC", "as_of_date": cutoff},
        },
    ]
    receipts = [
        {
            "tool": call["name"],
            "requested_cutoff": cutoff,
            "tool_cutoff": cutoff,
            "status": "passed",
        }
        for call in calls
    ]
    return {
        "final": {
            "matchup_id": game_id,
            "as_of_date": cutoff,
            "home_win_prob": 0.6,
            "away_win_prob": 0.4,
        },
        "tool_calls": calls,
        "gate_receipts": receipts,
        "tool_results": [
            {"name": call["name"], "status": "ok", "content": "{}"} for call in calls
        ],
        "system_prompt_sha256": "prompt-b",
    }


def test_agent_result_requires_a_passed_gate_receipt_for_every_required_call():
    result = _valid_agent_result()
    result["gate_receipts"] = []
    with pytest.raises(AssertionError, match="missing passed gate receipts"):
        assert_real_ui_result(
            result, "HOU-OKC-2025-10-21", "B", "2025-10-20", "prompt-b"
        )


def test_agent_result_rejects_duplicate_team_calls():
    result = _valid_agent_result()
    result["tool_calls"][2]["args"]["team_abbr"] = "HOU"
    with pytest.raises(AssertionError, match="wrong retrieve_team_form calls"):
        assert_real_ui_result(
            result, "HOU-OKC-2025-10-21", "B", "2025-10-20", "prompt-b"
        )


def test_agent_result_accepts_omitted_team_form_default_window():
    result = _valid_agent_result()
    for call in result["tool_calls"]:
        if call["name"] == "retrieve_team_form":
            call["args"].pop("last_n")

    assert_real_ui_result(
        result,
        "HOU-OKC-2025-10-21",
        "B",
        "2025-10-20",
        "prompt-b",
    )


def test_agent_result_rejects_a_required_tool_error():
    result = _valid_agent_result()
    result["tool_results"][0]["status"] = "error"
    with pytest.raises(AssertionError, match="required tool result unavailable"):
        assert_real_ui_result(
            result,
            "HOU-OKC-2025-10-21",
            "B",
            "2025-10-20",
            "prompt-b",
        )


def test_agent_result_rejects_malformed_required_tool_output():
    result = _valid_agent_result()
    result["tool_results"][0]["status"] = "invalid_json"
    with pytest.raises(AssertionError, match="required tool result unavailable"):
        assert_real_ui_result(
            result,
            "HOU-OKC-2025-10-21",
            "B",
            "2025-10-20",
            "prompt-b",
        )


def test_agent_result_accepts_opening_night_team_form_without_history():
    result = _valid_agent_result()
    for tool_result in result["tool_results"]:
        if tool_result["name"] == "retrieve_team_form":
            tool_result["status"] = "awaiting_input"
    assert_real_ui_result(
        result,
        "HOU-OKC-2025-10-21",
        "B",
        "2025-10-20",
        "prompt-b",
    )
