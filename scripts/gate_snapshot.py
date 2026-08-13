"""Materialise a date-gated copy of the data directory.

This is an optional extra physical copy for a single-date demo. It is not one
of the two runtime gates. Live requests and the full-season evaluator bind the
cutoff in ui/serve.py, then filter every read in agent/sources.py. Pointing
NBA_SNAPSHOT_DIR at a snapshot only changes which files those filters open.

WHY FUTURE GAME ROWS ARE KEPT. The agent is asked "who wins GAME X?" GAME X's
row is the question: teams and date. Dropping it would hide the matchup. The
answer lives in home_pts, away_pts, and winner, and those three columns are
blanked for any game after as_of. tests/test_snapshot_gate.py asserts both
halves: future rows exist, and they have no scores.

Injuries ARE gated here the same way injuries_as_of is: Date <= as_of. Rows
filed after the cutoff are dropped, not summarised.

    python -m scripts.gate_snapshot --as-of 2026-01-14
    NBA_SNAPSHOT_DIR=data/snapshots/2026-01-14 streamlit run ui/app.py

The two runtime filters still apply after this copy is built. A snapshot can
only be as strict as its loosest legitimate reader, so per-tool precision
(rest vs form) stays in agent/sources.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.sources import (  # noqa: E402
    INJURY_CSVS,
    ODDS_CSV,
    PLAYER_PER_GAME_CSV,
    PLAYER_STATS_CSV,
    RAW_DIR,
    SAMPLE_DIR,
    TEAM_SUMMARY_CSV,
    parse_date,
    season_end_year,
)

SNAPSHOT_ROOT = REPO_ROOT / "data" / "snapshots"

# Columns in game_logs_*.csv that state how a game turned out. That a game is
# scheduled is knowable in advance; how it ended is not. Future rows keep their
# identity and lose these.
OUTCOME_COLUMNS = ("home_pts", "away_pts", "winner")


def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def gate_game_logs(as_of: date, out_root: Path) -> list[dict]:
    """Keep every scheduled game; erase the result of any game after as_of.

    Dropping future rows outright would be the obvious move and it is wrong:
    the agent is asked to preview a game that has not been played, so it has to
    be able to see that the game exists. What it must not see is who won.
    """
    stats = []
    for src in sorted(SAMPLE_DIR.glob("game_logs_*.csv")):
        fieldnames, rows = _read(src)
        blanked = 0
        for row in rows:
            if parse_date(row["game_date"]) > as_of:
                if any(row.get(c) for c in OUTCOME_COLUMNS):
                    blanked += 1
                for col in OUTCOME_COLUMNS:
                    row[col] = ""
        _write(out_root / "samples" / src.name, fieldnames, rows)
        stats.append(
            {
                "file": f"samples/{src.name}",
                "rule": f"rows kept; outcome columns cleared where game_date > {as_of}",
                "rows_in": len(rows),
                "rows_out": len(rows),
                "outcomes_cleared": blanked,
            }
        )
    return stats


def gate_odds(as_of: date, out_root: Path) -> dict:
    """Drop any line for a game at or after as_of.

    The closing line is the market's final number and is not knowable until
    tip-off, so it cannot sit in the agent's snapshot at all. `eval/replay.py`
    reads the unfiltered file directly -- it is the benchmark holder and is
    allowed to know the answer.
    """
    fieldnames, rows = _read(ODDS_CSV)
    kept = [r for r in rows if parse_date(r["date"]) < as_of]
    _write(out_root / "samples" / ODDS_CSV.name, fieldnames, kept)
    return {
        "file": f"samples/{ODDS_CSV.name}",
        "rule": f"rows dropped where date >= {as_of} (closing line is not an as-of fact)",
        "rows_in": len(rows),
        "rows_out": len(kept),
        "outcomes_cleared": 0,
    }


def gate_injuries(as_of: date, out_root: Path) -> list[dict]:
    """Drop any injury report filed after as_of.

    Inclusive of as_of itself, matching `injuries_as_of`, which keeps records
    with `Date <= as_of`. A snapshot stricter than the query-time filter would
    silently change published results.
    """
    stats = []
    for src in INJURY_CSVS:
        if not src.exists():
            continue
        fieldnames, rows = _read(src)
        kept = [r for r in rows if parse_date(r["Date"]) <= as_of]
        rel = src.relative_to(RAW_DIR)
        _write(out_root / "raw" / rel, fieldnames, kept)
        stats.append(
            {
                "file": f"raw/{rel}",
                "rule": f"rows dropped where Date > {as_of}",
                "rows_in": len(rows),
                "rows_out": len(kept),
                "outcomes_cleared": 0,
            }
        )
    return stats


def gate_season_tables(as_of: date, out_root: Path) -> list[dict]:
    """Keep only seasons that had finished before the as-of season began.

    These are end-of-season aggregates, so the current season's row summarises
    games that have not been played yet. `player_importance` and `team_ratings`
    only ever read `season_end_year(as_of) - 1`, so cutting here costs nothing
    and removes a whole class of leak.
    """
    cutoff = season_end_year(as_of)
    stats = []
    for src in (TEAM_SUMMARY_CSV, PLAYER_PER_GAME_CSV):
        if not src.exists():
            continue
        fieldnames, rows = _read(src)
        kept = [r for r in rows if r.get("season") and int(r["season"]) < cutoff]
        rel = src.relative_to(RAW_DIR)
        _write(out_root / "raw" / rel, fieldnames, kept)
        stats.append(
            {
                "file": f"raw/{rel}",
                "rule": f"rows dropped where season >= {cutoff} (unfinished season)",
                "rows_in": len(rows),
                "rows_out": len(kept),
                "outcomes_cleared": 0,
            }
        )
    return stats


def gate_player_history(as_of: date, out_root: Path) -> dict:
    """Keep only completed player rows observable by the snapshot date."""
    fieldnames, rows = _read(PLAYER_STATS_CSV)
    kept = [r for r in rows if parse_date(r["game_date"]) <= as_of]
    _write(out_root / "exports" / PLAYER_STATS_CSV.name, fieldnames, kept)
    return {
        "file": f"exports/{PLAYER_STATS_CSV.name}",
        "rule": f"rows dropped where game_date > {as_of}",
        "rows_in": len(rows),
        "rows_out": len(kept),
        "outcomes_cleared": 0,
    }


def build_snapshot(as_of: date, out_root: Path | None = None) -> Path:
    """Write a gated copy of the data directory and return where it landed."""
    out_root = out_root or SNAPSHOT_ROOT / as_of.isoformat()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    files = [
        *gate_game_logs(as_of, out_root),
        gate_odds(as_of, out_root),
        *gate_injuries(as_of, out_root),
        *gate_season_tables(as_of, out_root),
        gate_player_history(as_of, out_root),
    ]

    manifest = {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(REPO_ROOT / "data"),
        "files": files,
        "note": (
            "Structural gate. The agent reads only this directory when "
            "NBA_SNAPSHOT_DIR points at it. Query-time filters in "
            "agent/sources.py still apply and are stricter per-tool."
        ),
    }
    (out_root / "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", required=True, help="cutoff date, YYYY-MM-DD")
    ap.add_argument("--out", help="destination (default data/snapshots/<as-of>)")
    args = ap.parse_args()

    as_of = parse_date(args.as_of)
    out = build_snapshot(as_of, Path(args.out) if args.out else None)

    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    # --out may point anywhere -- a tmp dir in the test suite, a scratch disk.
    # relative_to raises rather than falling back, so ask before shortening.
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"Snapshot as of {as_of} -> {shown}\n")
    for f in manifest["files"]:
        dropped = f["rows_in"] - f["rows_out"]
        detail = f"{f['rows_out']:>7,} kept"
        if dropped:
            detail += f"  {dropped:>7,} dropped"
        if f["outcomes_cleared"]:
            detail += f"  {f['outcomes_cleared']:>7,} outcomes cleared"
        print(f"  {f['file']:<46} {detail}")
    print(f"\nPoint the agent at it:\n  export NBA_SNAPSHOT_DIR={out}")


if __name__ == "__main__":
    main()
