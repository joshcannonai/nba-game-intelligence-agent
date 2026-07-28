"""Build the 2025-26 evaluation set from the raw odds file.

This is the season we test on (team decision, 2026-07-21 sync): regular season
for context, the playoffs as the held-out test set.

The raw file keeps score_away/score_home in the SAME ROW as the betting line.
Handing that row to a retrieval tool would hand the agent the final score. So
this script splits it into two files that cannot leak into each other:

    game_logs_2026.csv   schedule + results   (the answer key -- eval only)
    odds_2026.csv        the market's price   (no scores, ever)

`retrieve_betting_line` reads the second one. Nothing the agent can call reads
the first. Per the advisor (2026-07-21), the line is an evaluation baseline,
not a model input -- keeping the files apart makes that structural rather than
a matter of remembering.

    python scripts/build_2026_testset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/odds/primary/nba_2008-2026.csv"
OUT_GAMES = ROOT / "data/samples/game_logs_2026.csv"
OUT_ODDS = ROOT / "data/samples/odds_2026.csv"
SEASON = "2026"

# odds-source code -> the abbreviation our game logs use
TEAM_MAP = {
    "atl": "ATL",
    "bkn": "BRK",
    "bos": "BOS",
    "cha": "CHO",
    "chi": "CHI",
    "cle": "CLE",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gs": "GSW",
    "hou": "HOU",
    "ind": "IND",
    "lac": "LAC",
    "lal": "LAL",
    "mem": "MEM",
    "mia": "MIA",
    "mil": "MIL",
    "min": "MIN",
    "no": "NOP",
    "ny": "NYK",
    "okc": "OKC",
    "orl": "ORL",
    "phi": "PHI",
    "phx": "PHO",
    "por": "POR",
    "sa": "SAS",
    "sac": "SAC",
    "tor": "TOR",
    "utah": "UTA",
    "wsh": "WAS",
}


def truthy(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"missing {SRC} -- the raw odds file is gitignored, see README"
        )

    rows = [r for r in csv.DictReader(open(SRC)) if r["season"] == SEASON]
    games, odds, skipped = [], [], 0

    for r in sorted(rows, key=lambda x: x["date"]):
        home, away = TEAM_MAP.get(r["home"]), TEAM_MAP.get(r["away"])
        if not home or not away:
            skipped += 1
            continue

        date = r["date"]
        matchup_id = f"{away}-{home}-{date}"
        playoffs = truthy(r["playoffs"])

        hp, ap = r["score_home"], r["score_away"]
        if hp and ap:
            games.append(
                {
                    "game_id": matchup_id,
                    "game_date": date,
                    "home": home,
                    "away": away,
                    "home_pts": hp,
                    "away_pts": ap,
                    "winner": home if int(hp) > int(ap) else away,
                    "playoffs": "1" if playoffs else "0",
                }
            )

        # deliberately no score columns -- see module docstring
        odds.append(
            {
                "matchup_id": matchup_id,
                "game_date": date,
                "home": home,
                "away": away,
                "playoffs": "1" if playoffs else "0",
                "whos_favored": r["whos_favored"],
                "spread": r["spread"],
                "total": r["total"],
                "moneyline_home": r["moneyline_home"],
                "moneyline_away": r["moneyline_away"],
            }
        )

    OUT_GAMES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GAMES, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(games[0].keys()))
        w.writeheader()
        w.writerows(games)
    with open(OUT_ODDS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(odds[0].keys()))
        w.writeheader()
        w.writerows(odds)

    po = sum(1 for g in games if g["playoffs"] == "1")
    print(
        f"{OUT_GAMES.name}: {len(games)} games ({po} playoff), {games[0]['game_date']} -> {games[-1]['game_date']}"
    )
    print(f"{OUT_ODDS.name}:   {len(odds)} rows, no score columns")
    if skipped:
        print(f"skipped {skipped} rows with unmapped team codes")


if __name__ == "__main__":
    main()
