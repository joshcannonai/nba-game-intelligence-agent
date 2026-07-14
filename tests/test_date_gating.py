"""Leakage tests.

The project's whole claim is that the agent only ever sees what was knowable on
as_of_date. These tests are what make that claim checkable instead of asserted.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from agent.sources import (
    STALE_INJURY_DAYS,
    CsvSource,
    MockSource,
    get_source,
    injuries_as_of,
    injury_data_through,
    parse_matchup_id,
    season_end_year,
)
from agent.tools import build_tools

# A real matchup inside the injury log's coverage (log ends 2025-01-12).
REAL_MATCHUP = "LAL-BOS-2024-12-25"
REAL_AS_OF = "2024-12-24"


def test_season_end_year_splits_on_august():
    assert season_end_year(date(2024, 12, 20)) == 2025  # Dec -> 2024-25 season
    assert season_end_year(date(2025, 3, 5)) == 2025  # Mar -> same season
    assert season_end_year(date(2025, 10, 1)) == 2026  # Oct -> next season


def test_parse_matchup_id():
    away, home, game_date = parse_matchup_id("LAL-BOS-2026-01-15")
    assert (away, home, game_date) == ("LAL", "BOS", date(2026, 1, 15))


def test_parse_matchup_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_matchup_id("LAL vs BOS")


def test_injuries_never_include_a_future_publish_date():
    as_of = date(2024, 12, 24)
    for team in ("LAL", "BOS", "GSW", "POR"):
        for inj in injuries_as_of(team, as_of):
            assert inj["published"] <= as_of.isoformat(), (
                f"leak: {inj['player']} published {inj['published']} > as_of {as_of}"
            )


def test_injury_list_grows_monotonically_with_as_of():
    """Moving as_of earlier can only remove knowledge, never add it."""
    early = injuries_as_of("LAL", date(2023, 11, 1))
    late = injuries_as_of("LAL", date(2024, 12, 24))
    early_pubs = {i["published"] for i in early}
    assert all(p <= "2023-11-01" for p in early_pubs)
    assert isinstance(late, list)


def test_returned_player_is_not_still_listed_out():
    """A player re-acquired before as_of must drop off the out-list."""
    rows_through = injury_data_through()
    assert rows_through is not None, "injury CSV missing -- run git pull"
    # Anyone still listed out must have no 'back' record after their 'out' date;
    # the replay guarantees this, so the out-list must never contain duplicates.
    out = injuries_as_of("BOS", date(2024, 12, 24))
    names = [i["player"] for i in out]
    assert len(names) == len(set(names))


def test_departed_players_are_not_reported_as_injured():
    """The log records IL moves but never departures.

    Kemba Walker was relinquished by Boston in 2021, never 'acquired' back, and
    left the team. A naive replay still lists him out for BOS in 2024. He must
    not appear -- and no 'injury' may be older than the staleness window.
    """
    out = injuries_as_of("BOS", date(2024, 12, 24))
    names = {i["player"] for i in out}
    assert "Kemba Walker" not in names
    assert "Isaiah Thomas" not in names
    for inj in out:
        assert inj["days_out"] <= STALE_INJURY_DAYS


def test_injury_counts_are_plausible():
    """A team should have a handful out on a given night, not dozens."""
    for team in ("LAL", "BOS", "GSW", "DEN"):
        assert len(injuries_as_of(team, date(2024, 12, 24))) <= 8


def test_real_source_refuses_as_of_after_tipoff():
    """Asking as-of the day AFTER the game would leak the result."""
    src = CsvSource()
    with pytest.raises(ValueError, match="leak"):
        src.matchup_context(REAL_MATCHUP, "2024-12-26")


def test_real_source_uses_prior_completed_season_for_ratings():
    """Current-season aggregates contain post-as_of games, so we must not use them."""
    ctx = CsvSource().matchup_context(REAL_MATCHUP, REAL_AS_OF)
    # 2024-12-25 is in the 2025 season -> ratings must come from 2024.
    assert "2023-24 final" in ctx["home_team"]["basis"]
    assert "2023-24 final" in ctx["away_team"]["basis"]


def test_real_source_reports_missing_schedule_instead_of_guessing():
    """No game logs on main yet -> nulls with a reason, never a made-up number."""
    ctx = CsvSource().matchup_context(REAL_MATCHUP, REAL_AS_OF)
    rest = ctx["rest"]
    if rest.get("unavailable"):
        assert rest["home_days_rest"] is None
        assert rest["away_back_to_back"] is None
        assert ctx["h2h_last_5"] == []
    else:  # game logs were fetched; then they must be date-gated
        for g in ctx["h2h_last_5"]:
            assert g["date"] <= REAL_AS_OF


def test_rest_is_schedule_based_not_as_of_gated():
    """Rest comes from the schedule (published in August), so it must not change
    with as_of. Results DO change with as_of -- that is the line we are drawing."""
    src = CsvSource()
    early = src.matchup_context(REAL_MATCHUP, "2024-11-01")
    late = src.matchup_context(REAL_MATCHUP, REAL_AS_OF)
    assert early["rest"] == late["rest"]
    assert early["rest"]["home_days_rest"] == 1  # BOS played 12/23, game is 12/25


def test_h2h_results_are_gated_even_though_rest_is_not():
    """The complement of the test above: outcomes never precede as_of."""
    ctx = CsvSource().matchup_context(REAL_MATCHUP, REAL_AS_OF)
    for g in ctx["h2h_last_5"]:
        assert g["date"] <= REAL_AS_OF


def test_mock_source_still_gates_dates():
    ctx = MockSource().matchup_context("LAL-BOS-2026-01-15", "2026-01-10")
    for inj in ctx["injuries"]:
        assert inj["published"] <= "2026-01-10"
    for g in ctx["h2h_last_5"]:
        assert g["date"] <= "2026-01-10"


def test_mock_hides_an_injury_published_after_as_of():
    """The fixture's only injury is published 2026-01-14; as-of 2026-01-13 hides it."""
    hidden = MockSource().matchup_context("LAL-BOS-2026-01-15", "2026-01-13")
    shown = MockSource().matchup_context("LAL-BOS-2026-01-15", "2026-01-14")
    assert hidden["injuries"] == []
    assert len(shown["injuries"]) == 1


@pytest.mark.parametrize("kind", ["mock", "real"])
def test_tools_return_valid_json(kind):
    from agent.sources import get_source

    source = get_source(kind)
    matchup, as_of = (
        ("LAL-BOS-2026-01-15", "2026-01-14")
        if kind == "mock"
        else (REAL_MATCHUP, REAL_AS_OF)
    )
    tools = {t.name: t for t in build_tools(source)}
    ctx = json.loads(
        tools["retrieve_matchup_context"].invoke(
            {"matchup_id": matchup, "as_of_date": as_of}
        )
    )
    assert ctx["as_of_date"] == as_of
    assert ctx["source"] == kind

    pred = json.loads(
        tools["predict_win_probability"].invoke(
            {"home_abbr": "BOS", "away_abbr": "LAL", "as_of_date": as_of}
        )
    )
    assert 0.0 <= pred["home_win_prob"] <= 1.0
    assert pred["home_win_prob"] + pred["away_win_prob"] == pytest.approx(1.0)


def test_real_player_splits_admit_missing_b2b_instead_of_inventing():
    out = CsvSource().player_splits("LeBron James", back_to_back=True)
    assert out["pts_avg"] is not None
    assert out["b2b_pts_avg"] is None
    assert "b2b_unavailable" in out


# --- the tool interface is the contract with the team -----------------------

EXPECTED_TOOLS = {
    "retrieve_matchup_context",
    "retrieve_player_splits",
    "retrieve_schedule",
    "retrieve_team_form",
    "retrieve_injuries",
    "retrieve_news",
    "retrieve_betting_line",
    "predict_win_probability",
    "predict_stat_line",
    "predict_best_player",
}


@pytest.mark.parametrize("kind", ["mock", "real"])
def test_every_agreed_tool_exists(kind):
    """The whole surface is present NOW, so the data layer can drop in behind it."""
    names = {t.name for t in build_tools(get_source(kind))}
    assert EXPECTED_TOOLS <= names, f"missing: {EXPECTED_TOOLS - names}"


@pytest.mark.parametrize("kind", ["mock", "real"])
def test_unbuilt_tools_say_so_and_name_an_owner(kind):
    """A placeholder must never fabricate. It must announce itself and name a human.

    This is the guard against the failure mode where an unimplemented tool quietly
    returns an empty list and the agent reports 'nobody is injured'.
    """
    args = {
        "matchup_id": "LAL-BOS-2024-12-25",
        "as_of_date": "2024-12-24",
        "team_abbr": "BOS",
        "home_abbr": "BOS",
        "away_abbr": "LAL",
        "player_name": "Jayson Tatum",
    }
    for t in build_tools(get_source(kind)):
        payload = json.loads(t.invoke({k: v for k, v in args.items() if k in t.args}))
        if payload.get("status") == "not_implemented":
            assert payload.get("owner"), f"{t.name} is unbuilt but names no owner"
            assert payload.get("needs"), f"{t.name} is unbuilt but says what it needs"


def test_injuries_past_the_end_of_the_log_warn_rather_than_report_nobody_hurt():
    """The log stops 2025-01-12. Past that, 'no injuries' is a LIE, not a fact."""
    src = get_source("real")
    end = injury_data_through()
    after = (end + timedelta(days=30)).isoformat()
    payload = src.injuries("BOS", after)
    assert payload["injuries"] == [] or payload.get("warnings")
    assert payload.get("warnings"), "must warn that injuries are unknown, not zero"
    assert "UNKNOWN" in payload["warnings"][0]


def test_injuries_admit_they_carry_no_measure_of_player_importance():
    """Six bench players and one MVP must not look identical without saying so."""
    payload = get_source("real").injuries("LAL", "2024-12-24")
    assert "importance_unavailable" in payload
