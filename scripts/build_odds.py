"""Join the historical betting lines onto our game logs.

The odds source (Kaggle, CC0) uses its own team codes and carries no game id, so we
join on (date, home, away) after normalising abbreviations. Output is one row per
game with the market's price AND the actual result, which is what the eval harness
needs to ask: did we beat the line?

    python scripts/build_odds.py            -> data/samples/odds.csv

Note on what the market publishes: the moneyline is a *price* (two of them, one per
side). Converting both to probabilities and summing gives the overround -- the amount
by which the book's implied probabilities exceed 100%. That excess is the house edge,
and it is present on every line the book has ever posted.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ODDS_CSV = ROOT / "data/raw/odds/primary/nba_2008-2026.csv"
OUT = ROOT / "data/samples/odds.csv"

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


def implied_prob(american: pd.Series) -> np.ndarray:
    """American moneyline -> the probability the book's price implies."""
    ml = pd.to_numeric(american, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logs = pd.concat(
        [
            pd.read_csv(f)
            for f in sorted(glob.glob(str(ROOT / "data/samples/game_logs_*.csv")))
        ],
        ignore_index=True,
    )
    odds = pd.read_csv(ODDS_CSV)
    odds["home"] = odds.home.map(TEAM_MAP)
    odds["away"] = odds.away.map(TEAM_MAP)
    odds = odds.rename(columns={"date": "game_date"})

    m = logs.merge(
        odds[
            [
                "game_date",
                "home",
                "away",
                "whos_favored",
                "spread",
                "total",
                "moneyline_home",
                "moneyline_away",
            ]
        ],
        on=["game_date", "home", "away"],
        how="inner",
    )

    # The market's own numbers, made explicit.
    m["p_home_raw"] = implied_prob(m.moneyline_home)
    m["p_away_raw"] = implied_prob(m.moneyline_away)
    m["overround"] = m.p_home_raw + m.p_away_raw
    # De-vigged: strip the house edge out so the market's *opinion* can be scored
    # fairly against ours. Without this we would be marking the book down for
    # charging a fee, which is not a forecasting error.
    m["p_home_fair"] = m.p_home_raw / m.overround
    m["p_away_fair"] = m.p_away_raw / m.overround

    # Actual result, and whether the favourite covered.
    m["home_won"] = (m.home_pts > m.away_pts).astype(int)
    m["home_margin"] = m.home_pts - m.away_pts
    fav_home = m.whos_favored.eq("home")
    m["home_spread"] = np.where(fav_home, -m.spread, m.spread)
    m["home_covered"] = (m.home_margin + m.home_spread > 0).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False)

    if args.quiet:
        return
    ml = m.dropna(subset=["overround"])
    print(f"game logs        : {len(logs):,}")
    print(f"joined to odds   : {len(m):,}  ({len(m) / len(logs):.1%} of game logs)")
    print(f"  with spread    : {m.spread.notna().sum():,}")
    print(f"  with moneyline : {len(ml):,}")
    print(f"seasons          : {m.game_date.min()} .. {m.game_date.max()}")
    print()
    print("THE HOUSE EDGE, measured:")
    print(
        f"  mean overround        : {ml.overround.mean():.4f}  ({(ml.overround.mean() - 1) * 100:.2f}% above fair)"
    )
    print(f"  games priced >100%    : {(ml.overround > 1).mean():.3%}")
    print(f"  games priced <=100%   : {(ml.overround <= 1).sum():,}")
    print(f"  best price ever offered: {ml.overround.min():.4f}")
    print()
    print("BASELINES the agent has to beat:")
    print(f"  home team wins            : {m.home_won.mean():.3%}")
    print(
        f"  favourite covers spread   : {1 - m.home_covered.mean() if False else m.home_covered.mean():.3%}  (home side)"
    )
    print("  break-even at -110        : 52.381%")
    print(
        f"  market (de-vigged) picks winner: {(ml.p_home_fair.round() == ml.home_won).mean():.3%}"
    )
    print(f"  -> wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
