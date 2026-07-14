"""Pull one season of game logs (schedule + results) into data/samples/.

The datasets on main are season aggregates -- they have no game-by-game
schedule, so rest, back-to-back, and head-to-head cannot be computed and the
eval harness has no games to score. This is a thin pull to unblock that: one
row per game, the minimum the agent's tools need.

    python scripts/fetch_game_logs.py --season 2025

Writes data/samples/game_logs_<season>.csv with columns:
    game_id, game_date, home, away, home_pts, away_pts, winner

Season is the END year (2025 = the 2024-25 season), matching nba_stats.

This is a sample so mock shapes match live data, not a replacement for the
data layer's real scrape. It hits stats.nba.com, which rate-limits and
sometimes blocks cloud IPs; if it fails, the tools degrade honestly (nulls with
a reason) rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "samples"

sys.path.insert(0, str(REPO_ROOT))

from agent.teams import normalize_abbr  # noqa: E402


def season_label(end_year: int) -> str:
    """2025 -> '2024-25', the format nba_api expects."""
    return f"{end_year - 1}-{str(end_year)[2:]}"


def fetch(end_year: int) -> list[dict]:
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        raise SystemExit(
            "nba_api not installed. Run: pip install -r requirements.txt"
        ) from None

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season_label(end_year),
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    rows = finder.get_normalized_dict()["LeagueGameFinderResults"]

    # nba_api returns one row per team per game. Fold each pair into one game.
    games: dict[str, dict] = {}
    for r in rows:
        gid = r["GAME_ID"]
        # MATCHUP is "LAL vs. BOS" (home) or "LAL @ BOS" (away).
        is_home = "vs." in r["MATCHUP"]
        team = normalize_abbr(r["TEAM_ABBREVIATION"])
        pts = r["PTS"]
        game = games.setdefault(gid, {"game_id": gid, "game_date": r["GAME_DATE"]})
        if is_home:
            game["home"], game["home_pts"] = team, pts
        else:
            game["away"], game["away_pts"] = team, pts

    complete = []
    for g in games.values():
        if not {"home", "away", "home_pts", "away_pts"} <= g.keys():
            continue  # half a game (preseason artifact / in-progress); skip it
        if g["home_pts"] is None or g["away_pts"] is None:
            continue
        g["winner"] = g["home"] if g["home_pts"] > g["away_pts"] else g["away"]
        complete.append(g)

    complete.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season END year (2025 = 2024-25 season). Default: 2025",
    )
    args = parser.parse_args()

    games = fetch(args.season)
    if not games:
        raise SystemExit(f"No games returned for {season_label(args.season)}.")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLE_DIR / f"game_logs_{args.season}.csv"
    fields = ["game_id", "game_date", "home", "away", "home_pts", "away_pts", "winner"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(games)

    print(
        f"{len(games)} games -> {out.relative_to(REPO_ROOT)} "
        f"({games[0]['game_date']} .. {games[-1]['game_date']})"
    )


if __name__ == "__main__":
    main()
