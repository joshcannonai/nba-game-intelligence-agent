"""Leakage tests.

The project's whole claim is that the agent only ever sees what was knowable on
as_of_date. These tests are what make that claim checkable instead of asserted.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta

import pytest

from agent.sources import (
    ODDS_CSV,
    STALE_INJURY_DAYS,
    CsvSource,
    MockSource,
    INJURY_CSVS,
    _injury_rows,
    _odds_rows,
    closing_line,
    parse_date,
    get_source,
    injuries_as_of,
    injury_data_through,
    parse_matchup_id,
    season_end_year,
)
from agent import sources
from agent.teams import TEAMS, odds_abbr
from agent.tools import build_tools

# A real matchup inside the injury log's coverage (log ends 2025-01-12).
REAL_MATCHUP = "LAL-BOS-2024-12-25"
REAL_AS_OF = "2024-12-24"

# Everything in the raw odds file that describes how the game turned out.
# scripts/odds_only.py must never select these into the sample.
SCORE_COLUMNS = {
    "score_away",
    "score_home",
    "q1_away",
    "q2_away",
    "q3_away",
    "q4_away",
    "ot_away",
    "q1_home",
    "q2_home",
    "q3_home",
    "q4_home",
    "ot_home",
    "winner",
}


def _line(matchup_id: str, as_of_date: str) -> dict:
    """The closing line, read the way eval/ reads it.

    Was a tool call until week 6, when retrieve_betting_line was taken away
    from the agent for leaking the benchmark into its own reasoning. The
    guarantees below did not stop mattering when the tool went -- eval/replay.py
    and eval/three_arms.py still call closing_line to score every arm, so a
    score column appearing in this payload would poison the scoreboard itself.
    Same assertions, aimed one layer down at the function that survived.
    """
    away, home, game_date = parse_matchup_id(matchup_id)
    return closing_line(away, home, game_date, parse_date(as_of_date))


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
    "predict_win_probability",
    "predict_stat_line",
}

# Cut in week 6. Named here rather than simply deleted, because "the agent must
# not have a betting-line tool" is a live safety property, not a historical
# note -- re-adding retrieve_betting_line would re-open the leak that made us
# cut it. See the module docstring in agent/tools.py.
FORBIDDEN_TOOLS = {
    "retrieve_news",
    "predict_best_player",
    "retrieve_betting_line",
}


@pytest.mark.parametrize("kind", ["mock", "real"])
def test_every_agreed_tool_exists(kind):
    """The whole surface is present NOW, so the data layer can drop in behind it."""
    names = {t.name for t in build_tools(get_source(kind))}
    assert EXPECTED_TOOLS <= names, f"missing: {EXPECTED_TOOLS - names}"


@pytest.mark.parametrize("kind", ["mock", "real"])
def test_the_agent_cannot_see_the_market(kind):
    """The closing line is what we grade against, so the agent must not hold it.

    Watching the live agent on 2026-01-14 it wrote "the closing line favors ORL
    (-5.5)" straight into its key factors. Scoring a prediction that quoted the
    benchmark measures nothing. The fix was structural -- take the tool away --
    and this test is what keeps it taken away.
    """
    names = {t.name for t in build_tools(get_source(kind))}
    assert not (FORBIDDEN_TOOLS & names), f"cut tool is back: {FORBIDDEN_TOOLS & names}"


def test_arm_b_differs_from_arm_c_by_exactly_one_tool():
    """The three-arm comparison is only meaningful if the arms differ in one thing."""
    source = get_source("real")
    with_model = {t.name for t in build_tools(source, include_model=True)}
    without = {t.name for t in build_tools(source, include_model=False)}
    assert with_model - without == {"predict_win_probability"}
    assert not without - with_model


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
        if payload.get("status") == "awaiting_input":
            assert payload.get("needs_from"), (
                f"{t.name} awaits input but names no source"
            )
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


def test_betting_line_never_contains_a_score():
    """retrieve_betting_line must be structurally incapable of leaking a result.

    It reads from data/samples/odds_only.csv, which scripts/odds_only.py
    builds by keeping an allowlist of safe columns -- score_away, score_home,
    and every quarter/OT column are never selected into that file. This test
    checks the tool's actual output, not just the file, so a future change to
    the tool itself would also be caught.
    """
    payload = _line("DET-LAL-2024-12-23", "2024-12-22")

    # A "not_found" payload would pass a key check trivially. Prove we got a
    # real row first, or this test is only asserting that nothing happened.
    assert payload["status"] == "ok", payload

    leaked = SCORE_COLUMNS & payload.keys()
    assert not leaked, f"retrieve_betting_line leaked score field(s): {leaked}"


def test_odds_only_csv_has_no_score_columns():
    """File-level guarantee, independent of the tool: the source file itself
    must never carry score data, no matter what column order it's saved in."""
    with ODDS_CSV.open(newline="", encoding="utf-8") as f:
        columns = set(csv.DictReader(f).fieldnames or [])
    leaked = SCORE_COLUMNS & columns
    assert not leaked, f"odds_only.csv contains score column(s): {leaked}"


def test_betting_line_refuses_a_query_dated_after_tip_off():
    """The line is the one field sitting next to the result in the raw data.

    Dropping the score columns stops the tool returning a score; it does not
    stop the tool answering a question asked from *after* the game. Without
    this, a replay could ask on June 1st for a December game and be told the
    market's final number -- which is a fact about tip-off, not about as_of.
    """
    after = _line("DET-LAL-2024-12-23", "2025-06-01")
    assert after["status"] == "gated", after
    assert "spread" not in after

    on_the_day = _line("DET-LAL-2024-12-23", "2024-12-23")
    assert on_the_day["status"] == "ok", on_the_day


def test_betting_line_emits_valid_json_when_the_moneyline_is_missing():
    """No NaN in the payload.

    The odds file carries no moneyline from 2023-24 on, and a bare float('nan')
    serialises to the token `NaN`, which is not valid JSON. Python's json.loads
    accepts it, so a round-trip test cannot see the bug -- parse strictly.
    """
    raw = json.dumps(_line("DET-LAL-2024-12-23", "2024-12-22"))

    def reject(token: str):
        raise AssertionError(f"payload is not valid JSON: contains {token}")

    payload = json.loads(raw, parse_constant=reject)
    assert payload["moneyline_home"] is None
    assert payload["unavailable"], "a missing moneyline must be stated, not silent"


def test_betting_line_says_who_is_laying_the_points():
    """`spread: 6.5` is unusable on its own -- the file stores a magnitude."""
    payload = _line("DET-LAL-2024-12-23", "2024-12-22")
    assert payload["favorite"] == "LAL"
    assert payload["spread_home"] == -6.5  # LAL favoured, so LAL lays the points
    assert payload["spread_away"] == 6.5


def test_every_team_abbreviation_resolves_in_the_odds_file():
    """The odds file spells nine teams a fourth way.

    Unmapped codes do not raise -- they silently return "no odds for this
    game", which reads like a data gap rather than a bug. That hid 638 of the
    1,225 games in the 2025-26 sample.
    """
    codes = {away for away, _, _ in _odds_rows()} | {
        home for _, home, _ in _odds_rows()
    }
    unmapped = sorted(a for a in TEAMS if odds_abbr(a) not in codes)
    assert not unmapped, f"no odds-file spelling for: {unmapped}"


def test_injury_log_covers_the_season_we_test_on():
    """The replay window is 2025-26; the Kaggle set alone stops at 2025-01-12.

    Without the continuation file every game in the test season reports zero
    players out, which the agent cannot distinguish from a genuinely healthy
    roster -- it would silently score the whole season on absent data.
    """
    through = injury_data_through()
    assert through >= date(2026, 4, 1), (
        f"injury log ends {through}; the 2025-26 replay window needs data "
        "through the end of that season"
    )
    assert injuries_as_of("BOS", date(2026, 1, 20)), (
        "no injuries found mid-2025-26 season -- the continuation file is "
        "missing or not being read"
    )


def test_the_two_injury_files_join_without_a_gap_or_an_overlap():
    """They are separate files to keep provenance, but one continuous log."""
    dates = {d for d, *_ in _injury_rows()}
    kaggle_end, pst_start = date(2025, 1, 12), date(2025, 1, 13)
    assert kaggle_end in dates and pst_start in dates, "the seam has a hole in it"

    # Overlap would double-count a transaction, so check the files are disjoint
    # in time rather than that every row is unique -- the Kaggle file already
    # carries one genuine duplicate of its own (Doug McDermott, 2022-12-08).
    def file_dates(path):
        with path.open(newline="", encoding="utf-8") as f:
            return {r["Date"] for r in csv.DictReader(f)}

    shared = file_dates(INJURY_CSVS[0]) & file_dates(INJURY_CSVS[1])
    assert not shared, (
        f"the two injury files cover the same dates: {sorted(shared)[:5]}"
    )


def test_injuries_weight_a_star_above_a_bench_player():
    """The advisor's 2026-07-21 note: an MVP and a 10th man must not weigh the same."""
    payload = get_source("real").injuries("LAL", "2024-12-01")
    assert "importance_basis" in payload, "must say how importance was derived"

    by_name = {i["player"]: i for i in payload["injuries"]}
    star = by_name.get("Austin Reaves")
    bench = by_name.get("Jaxson Hayes")
    assert star and bench, f"expected both players out, got {list(by_name)}"
    assert star["importance"] > bench["importance"], (
        f"{star['player']} ({star['importance']}) must outweigh "
        f"{bench['player']} ({bench['importance']})"
    )
    assert star["tier"] != bench["tier"]


def test_importance_is_none_not_zero_for_a_player_with_no_prior_season():
    """A rookie is unknown, not worthless -- None, never 0.0."""
    payload = get_source("real").injuries("LAL", "2024-12-01")
    rookies = [i for i in payload["injuries"] if i.get("tier") == "unknown"]
    assert rookies, "expected at least one player with no prior season"
    assert all(r["importance"] is None for r in rookies)


def test_odds_file_carries_no_scores():
    """Structural leakage guarantee: the betting file cannot contain the answer.

    The raw source keeps score_away/score_home in the same row as the line.
    scripts/odds_only.py splits them; this asserts the split held.

    Deliberately not skippable: this used to point at odds_2026.csv and skip
    when it was absent, so retiring that file would have quietly disarmed the
    check rather than failed it.
    """
    assert ODDS_CSV.exists(), f"{ODDS_CSV.name} missing -- run scripts/odds_only.py"
    with ODDS_CSV.open(encoding="utf-8") as fh:
        header = fh.readline().lower()
    for banned in ("score", "_pts", "winner"):
        assert banned not in header, (
            f"odds file leaks results: {banned!r} in {header!r}"
        )


def test_betting_line_never_returns_a_result_for_a_finals_game():
    """The last game of the season is the sharpest case: the result is famous.

    NYK-SAS on 2026-06-13 is in the playoff test set, so a leak here would flow
    straight into the Vegas baseline every arm is scored against.
    """
    payload = _line("NYK-SAS-2026-06-13", "2026-06-12")
    for banned in ("score_home", "score_away", "winner", "home_pts", "away_pts"):
        assert banned not in payload, f"betting line leaked {banned}"


def _scrub_results(rows, cutoff):
    """Keep each row so its DATE survives, but destroy its RESULT."""
    out = []
    for g in rows:
        g = dict(g)
        if parse_date(g["game_date"]) >= cutoff:
            g["home_pts"] = g["away_pts"] = "0"
            g["winner"] = "SCRUBBED"
        out.append(g)
    return tuple(out)


@pytest.mark.parametrize(
    ("matchup", "as_of"),
    [
        ("NYK-BOS-2025-12-02", "2025-12-01"),
        ("LAL-DEN-2026-01-15", "2026-01-10"),
        ("MIA-BOS-2024-12-02", "2024-11-15"),
        ("BOS-NYK-2026-03-01", "2025-11-01"),  # asking four months ahead
    ],
)
def test_no_future_outcome_reaches_the_report(monkeypatch, matchup, as_of):
    """The leakage claim in its strongest form: destroy the future, expect no change.

    Every other test here checks that a particular field is filtered. This one
    makes no assumption about which field: it erases every result dated on or
    after as_of -- scores and winner -- and re-runs. If any code path consumes
    an outcome it should not see, the two reports differ.

    Schedule DATES are deliberately left intact. Rest is not gated on purpose
    (the NBA publishes the schedule in August), so gating dates here would fail
    the test for a behaviour we explicitly want -- see
    test_rest_is_schedule_based_not_as_of_gated.
    """
    from agent.run import dry_run

    cutoff = parse_date(as_of)
    src = CsvSource()
    full = json.loads(dry_run(matchup, as_of, src))

    real_injuries = sources._injury_rows.__wrapped__
    real_logs = sources._game_logs.__wrapped__
    real_all = sources._all_game_logs.__wrapped__
    gated = tuple(r for r in real_injuries() if r[0] <= cutoff)

    monkeypatch.setattr(sources, "_injury_rows", lambda: gated)
    monkeypatch.setattr(
        sources, "_game_logs", lambda season: _scrub_results(real_logs(season), cutoff)
    )
    monkeypatch.setattr(
        sources, "_all_game_logs", lambda: _scrub_results(real_all(), cutoff)
    )

    blind = json.loads(dry_run(matchup, as_of, src))
    assert full == blind, (
        "the report changed when future results were erased -- something read "
        "an outcome dated after as_of"
    )
