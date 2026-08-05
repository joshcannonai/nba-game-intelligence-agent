"""What is a top scorer's absence actually worth?

This exists to answer a question the team left open on 2026-08-03: when a player
averaging more than 20 ppg is out, by how much should the win probability drop?
We wanted a number to put in the agent's rules. The honest answer, on our data,
is that there is no number to put there, and the way we got to that matters more
than the result.

Three comparisons, each one correcting the previous one's flaw:

  1. NAIVE       every game, split on whether either side's star was out.
                 Says the home team wins LESS when the AWAY star is out, which
                 is backwards, so something is wrong.

  2. CONDITIONED restrict to teams that HAVE a >20 ppg player, then split on
                 whether he was out. Still says teams win MORE without him
                 (+5.6%, z = +2.6). Also backwards, and now significant.

                 Both are confounded the same way. Having a 20 ppg scorer is a
                 property of good teams, and a good team is still good on the
                 night its star sits. The split is comparing strong rosters to
                 weak ones, not healthy lineups to depleted ones.

  3. WITHIN-TEAM compare each team against ITSELF -- its win rate with the star
                 against its win rate without him -- then average the per-team
                 differences. Team quality cancels because it appears on both
                 sides. This is the only one of the three that answers the
                 question asked.

    python -m eval.injury_impact
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources import (  # noqa: E402
    PLAYER_PER_GAME_CSV,
    SAMPLE_DIR,
    injuries_as_of,
    parse_date,
    season_end_year,
)
from agent.teams import normalize_abbr  # noqa: E402

TOP_PPG = 20.0
MIN_GAMES = 8  # a team must have this many games in BOTH states to contribute
MULTI_TEAM = {"2TM", "3TM", "4TM", "5TM", "TOT"}


def star_rosters() -> dict[tuple[int, str], set[str]]:
    """(season, team) -> players who averaged over TOP_PPG that season.

    Multi-team rows are dropped: a player traded mid-season has a combined line
    that belongs to no single roster.
    """
    rosters: dict[tuple[int, str], set[str]] = {}
    with PLAYER_PER_GAME_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                season = int(row["season"])
                pts = float(row["pts_per_game"] or 0)
            except (ValueError, KeyError):
                continue
            if pts > TOP_PPG and row["team"] not in MULTI_TEAM:
                key = (season, normalize_abbr(row["team"]))
                rosters.setdefault(key, set()).add(row["player"])
    return rosters


def team_games(season: int):
    """Yield (team_abbr, star_was_out, team_won) for every team-game."""
    rosters = star_rosters()
    path = SAMPLE_DIR / f"game_logs_{season}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for game in csv.DictReader(fh):
            as_of = parse_date(game["game_date"]) - timedelta(days=1)
            prior = season_end_year(as_of) - 1
            for side in ("home", "away"):
                abbr = normalize_abbr(game[side])
                roster = rosters.get((prior, abbr), set())
                if not roster:
                    continue  # no star to lose; the question does not apply
                out = {i["player"] for i in injuries_as_of(abbr, as_of)}
                yield abbr, bool(roster & out), game["winner"] == game[side]


def conditioned(rows) -> None:
    """Comparison 2: pooled across teams. Confounded -- shown to be dismissed."""
    tally = {True: [0, 0], False: [0, 0]}
    for _, star_out, won in rows:
        tally[star_out][0] += won
        tally[star_out][1] += 1

    (wa, na), (wb, nb) = tally[False], tally[True]
    if not na or not nb:
        return
    pa, pb = wa / na, wb / nb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    print("2. POOLED ACROSS TEAMS (confounded)")
    print(f"   star available  {pa:6.1%}  (n={na})")
    print(f"   star out        {pb:6.1%}  (n={nb})")
    print(f"   delta {pb - pa:+.1%}   se {se:.1%}   z {(pb - pa) / se:+.2f}")
    print("   Reads backwards. Teams with a star are better teams; this is")
    print("   measuring roster quality, not availability.\n")


def within_team(rows) -> None:
    """Comparison 3: each team against itself. The one that answers the question."""
    per_team: dict[str, dict[bool, list[int]]] = {}
    for abbr, star_out, won in rows:
        d = per_team.setdefault(abbr, {True: [0, 0], False: [0, 0]})
        d[star_out][0] += won
        d[star_out][1] += 1

    print("3. WITHIN TEAM (each team against itself)\n")
    print(f"   {'team':>5} {'with star':>17} {'without':>17} {'delta':>8}")
    deltas, weights = [], []
    for abbr in sorted(per_team):
        (wn, nn), (wo, no) = per_team[abbr][False], per_team[abbr][True]
        if nn < MIN_GAMES or no < MIN_GAMES:
            continue
        pw, po = wn / nn, wo / no
        deltas.append(po - pw)
        weights.append(min(nn, no))
        print(
            f"   {abbr:>5} {pw:>9.1%} (n={nn:>3}) {po:>9.1%} (n={no:>3}) {po - pw:>+8.1%}"
        )

    if not deltas:
        print("   (no team had enough games in both states)")
        return

    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    weighted = sum(d * w for d, w in zip(deltas, weights)) / sum(weights)
    worse = sum(d < 0 for d in deltas)

    print(f"\n   teams qualifying (>= {MIN_GAMES} games in both states): {len(deltas)}")
    print(f"   mean delta {mean:+.1%}   se {se:.1%}   z {mean / se if se else 0:+.2f}")
    print(f"   game-weighted {weighted:+.1%}")
    print(f"   teams that did worse without their star: {worse}/{len(deltas)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2026, help="Season end year")
    args = ap.parse_args()

    rows = list(team_games(args.season))
    print(
        f"team-games with a >{TOP_PPG:.0f} ppg player on the prior-season roster: "
        f"{len(rows)}\n"
    )
    conditioned(rows)
    within_team(rows)

    print(
        "\nCONCLUSION\n"
        "  No blanket percentage is supportable from this data. Within team the\n"
        "  effect is indistinguishable from zero and the spread across teams is\n"
        "  enormous, so a fixed 'star out -> drop N%' rule would be inventing a\n"
        "  number. The agent should defer to the fitted model's injury term\n"
        "  rather than apply a rule of its own. See skills/retrieve_injuries.md.\n"
        "\n"
        "  Caveat that limits all three numbers: star-to-team mapping comes from\n"
        "  the PRIOR season, so a player who changed teams over the summer still\n"
        "  counts against his old club. Some 'without star' samples are therefore\n"
        "  roster changes rather than injuries -- BOS shows 67 of 89 games\n"
        "  'without', which is a departure, not an absence. Fixing this needs a\n"
        "  current-season roster, which we do not have as an as-of source."
    )


if __name__ == "__main__":
    main()
