"""The structural gate: prove the snapshot directory never held the future.

`test_date_gating.py` proves the query-time filters refuse future records.
This proves something stronger and simpler to audit -- that the bytes on disk
do not contain the answer. The two are independent: if the filter regressed,
these tests would still pass, and vice versa. The last test here is the one
that ties them together.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

from agent.sources import REPO_ROOT, parse_date, season_end_year
from scripts.gate_snapshot import build_snapshot

AS_OF = "2026-01-14"

# Spelled out rather than imported from gate_snapshot. Importing the module's
# own OUTCOME_COLUMNS would make this file agree with the code under test by
# construction: emptying that constant would empty the assertion loop below
# and the leak test would pass while clearing nothing. Caught by mutation.
RESULT_COLUMNS = ("home_pts", "away_pts", "winner")


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    out = tmp_path_factory.mktemp("snap")
    return build_snapshot(parse_date(AS_OF), out)


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_no_game_result_after_as_of(snapshot):
    """A future game may appear on the schedule. It may not have a winner."""
    as_of = parse_date(AS_OF)
    leaked = []
    for path in sorted((snapshot / "samples").glob("game_logs_*.csv")):
        for row in _rows(path):
            if parse_date(row["game_date"]) > as_of:
                for col in RESULT_COLUMNS:
                    if row.get(col):
                        leaked.append((path.name, row["game_date"], col, row[col]))
    assert not leaked, (
        f"outcome of a future game survived into the snapshot: {leaked[:5]}"
    )


def test_future_games_are_still_visible(snapshot):
    """The complement of the test above: gating must not erase the schedule.

    Dropping the rows would also pass the leak test while making the agent
    unable to see that the game it is previewing exists at all.
    """
    as_of = parse_date(AS_OF)
    future = [
        r
        for r in _rows(snapshot / "samples" / "game_logs_2026.csv")
        if parse_date(r["game_date"]) > as_of
    ]
    assert future, (
        "every future game was dropped -- the agent cannot see its own matchup"
    )
    assert all(r["home"] and r["away"] for r in future)


def test_no_closing_line_at_or_after_as_of(snapshot):
    """The line is set at tip-off, so the agent may not hold one for as_of or later."""
    as_of = parse_date(AS_OF)
    late = [
        r["date"]
        for r in _rows(snapshot / "samples" / "odds_only.csv")
        if parse_date(r["date"]) >= as_of
    ]
    assert not late, (
        f"{len(late)} closing lines at/after as_of survived, first {late[:3]}"
    )


def test_no_injury_report_filed_after_as_of(snapshot):
    as_of = parse_date(AS_OF)
    late = []
    for path in sorted((snapshot / "raw").rglob("injury_data.csv")):
        late += [r["Date"] for r in _rows(path) if parse_date(r["Date"]) > as_of]
    assert not late, f"{len(late)} injury reports filed after as_of survived"


def test_no_unfinished_season_aggregates(snapshot):
    """End-of-season tables summarise games that have not been played yet."""
    cutoff = season_end_year(parse_date(AS_OF))
    late = []
    for path in sorted((snapshot / "raw").rglob("*.csv")):
        rows = _rows(path)
        if not rows or "season" not in rows[0]:
            continue
        late += [(path.name, r["season"]) for r in rows if int(r["season"]) >= cutoff]
    assert not late, f"unfinished-season rows survived: {late[:5]}"


def test_manifest_accounts_for_every_file(snapshot):
    """The manifest is the audit trail. It has to match what is on disk."""
    manifest = json.loads((snapshot / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["as_of"] == AS_OF
    listed = {f["file"] for f in manifest["files"]}
    on_disk = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*.csv")}
    assert listed == on_disk, f"manifest/disk mismatch: {listed ^ on_disk}"
    assert all(f["rule"] for f in manifest["files"]), "every file states its rule"


def _dry_run(matchup: str, as_of: str, snapshot_dir=None) -> dict:
    """Run the report in a fresh interpreter.

    Deliberately not `importlib.reload`: the data readers are `lru_cache`d and
    module-level path constants are captured at import, so an in-process reload
    can quietly keep reading the ungated directory and make the comparison
    below pass without proving anything.
    """
    env = dict(os.environ)
    env.pop("NBA_SNAPSHOT_DIR", None)
    if snapshot_dir is not None:
        env["NBA_SNAPSHOT_DIR"] = str(snapshot_dir)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.run",
            "--dry-run",
            "--source",
            "real",
            "--matchup",
            matchup,
            "--as-of",
            as_of,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_subprocess_actually_reads_the_snapshot(snapshot):
    """Guards the test below from passing for the wrong reason."""
    env = dict(os.environ, NBA_SNAPSHOT_DIR=str(snapshot))
    out = subprocess.run(
        [sys.executable, "-c", "import agent.sources as s; print(s.SAMPLE_DIR)"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(snapshot) in out.stdout, f"env var ignored, read {out.stdout!r}"


def test_snapshot_and_full_data_agree(snapshot):
    """The point of having both layers: they must reach the same answer.

    The snapshot physically cannot leak, so if the query-time filter were
    leaking these two would disagree. Equality is a check on the filter.
    """
    matchup = "MEM-ORL-2026-01-15"
    assert _dry_run(matchup, AS_OF, snapshot) == _dry_run(matchup, AS_OF)
