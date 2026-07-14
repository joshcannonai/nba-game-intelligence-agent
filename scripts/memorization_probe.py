"""Does the LLM already know who won? Measure it; do not trust the model card.

Date-gated retrieval closes ONE leakage channel: what we put in the context window.
It cannot touch the other one -- what the model memorised during training. A hosted
model that has read about the 2024-25 season can recall the result of a game we are
asking it to "predict", and our carefully gated pipeline would never notice.

So we probe it. Strip every tool away, give the model NO data, and ask it cold:

    "Lakers at Celtics, 2024-12-25. Who won?"

Then score accuracy season by season. Read it against three baselines:

    50.0%   coin flip -- knows nothing
    56.3%   always pick the home team (the real home-win rate in our data)
    66.9%   the de-vigged betting market
    >75%    it is recalling, not reasoning

A season where the model sits at the home-team baseline is a season it does not
remember, and is therefore SAFE TO TEST ON. A season well above it is contaminated.
The point where the curve falls off is the training cutoff -- measured, not assumed.

    python scripts/memorization_probe.py --per-season 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent.teams import full_name  # noqa: E402

ODDS = ROOT / "data/raw/odds/primary/nba_2008-2026.csv"
OUT = ROOT / "data/samples/memorization_probe.json"

# Same team-code mapping as build_odds.py.
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

PROMPT = """NBA regular season game.

{away} at {home}
Date: {date}

Which team won this game? This is a historical game with a known result.
Answer with exactly one word: HOME or AWAY. No explanation."""


def build_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY missing. Run ./scripts/set-key.sh google")
    # No tools. No retrieval. Weights only -- that is the entire point.
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"), temperature=0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-season", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sleep", type=float, default=1.2, help="free-tier rate limit")
    ap.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Only probe these seasons (default: all). 2026 = the 2025-26 season.",
    )
    args = ap.parse_args()

    d = pd.read_csv(ODDS).dropna(subset=["score_home", "score_away"])
    d = d[d.regular]
    d["home"] = d.home.map(TEAM_MAP)
    d["away"] = d.away.map(TEAM_MAP)
    d = d.dropna(subset=["home", "away"])
    d["home_won"] = d.score_home > d.score_away

    rng = random.Random(args.seed)
    model = build_model()
    results: list[dict] = []

    seasons = args.seasons or sorted(d.season.unique())
    for season in seasons:
        pool = d[d.season == season]
        if pool.empty:
            continue
        idx = rng.sample(range(len(pool)), min(args.per_season, len(pool)))
        games = pool.iloc[idx]
        correct = hedged = failed = answered = 0
        for _, g in games.iterrows():
            q = PROMPT.format(
                away=full_name(g.away) or g.away,
                home=full_name(g.home) or g.home,
                date=g.date,
            )
            try:
                raw = model.invoke(q).content
            except Exception as e:
                # A failed call is NOT a wrong answer. Count it separately or the
                # accuracy denominator silently absorbs it and every quota-blocked
                # season reports 0% -- which reads exactly like "the model is always
                # wrong" and is completely false. (Caught 2026-07-13, the hard way.)
                failed += 1
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    print(
                        f"  [{season}] QUOTA EXHAUSTED after {answered} answered. "
                        f"Free tier on this model is tiny -- use a local model.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                print(f"  [{season}] error: {msg[:70]}", file=sys.stderr)
                time.sleep(5)
                continue

            ans = str(raw).strip().upper()
            pick_home = "HOME" in ans and "AWAY" not in ans
            pick_away = "AWAY" in ans and "HOME" not in ans
            if not (pick_home or pick_away):
                hedged += 1
                continue
            answered += 1
            if pick_home == bool(g.home_won):
                correct += 1
            time.sleep(args.sleep)

        if answered == 0:
            print(
                f"season {season}  NO DATA (hedged {hedged}, failed {failed}) -- skipping",
                flush=True,
            )
            continue
        acc = correct / answered
        base = float(games.home_won.mean())
        results.append(
            {
                "season": int(season),
                "n": answered,
                "hedged": hedged,
                "failed": failed,
                "accuracy": round(acc, 4),
                "home_rate": round(base, 4),
            }
        )
        print(
            f"season {season}  n={answered:2d}  accuracy {acc:6.1%}   "
            f"(always-home would score {base:.1%})",
            flush=True,
        )

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT.relative_to(ROOT)}")

    hot = [r for r in results if r["accuracy"] >= 0.75]
    cold = [r for r in results if r["accuracy"] <= r["home_rate"] + 0.05]
    print(
        "\nCONTAMINATED (>=75%, the model is recalling):",
        ", ".join(str(r["season"]) for r in hot) or "none",
    )
    print(
        "SAFE TO TEST ON (at/below the always-home baseline):",
        ", ".join(str(r["season"]) for r in cold) or "none",
    )


if __name__ == "__main__":
    main()
