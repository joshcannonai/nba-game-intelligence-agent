"""The feature vector for one game, built only from what was knowable that morning.

This file is the reason the model is allowed to exist alongside the agent. Both
of them answer the same question, so if the model got to see anything the agent
could not, the comparison would be rigged and the whole three-arm experiment
would be measuring our plumbing instead of our ideas.

Two gates run in series, same as the agent's:

    the snapshot   data/snapshots/<as_of>/ physically holds nothing later
    these features every value below is derived from games strictly BEFORE the
                   game being predicted

The second gate is the one that is easy to get wrong, so it is worth naming the
trap: a team's season win percentage is a legitimate feature, but the obvious
way to compute it -- group the season and take the mean -- silently includes the
game you are predicting. That is a 100%-accurate model and a worthless one.
Every accumulator here is advanced AFTER the row is emitted, never before.

Rest is the one thing measured in schedule dates rather than results. The NBA
publishes its full schedule in August, so "BOS plays the 23rd and the 25th" is
knowable on any as-of date; only the outcome is gated. `agent.sources` makes the
same distinction in `schedule_context`, deliberately.

Speed matters here -- 3,777 games times two teams -- so this walks each season
once and carries the accumulators forward, rather than re-scanning the log per
game the way `agent.sources.team_form_as_of` does. That makes it a second
implementation of the same idea, which is a risk, so
`tests/test_model_contract.py` asserts the two agree game for game. If they ever
diverge, that test fails rather than the model quietly training on a different
world than the agent sees.
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from agent.sources import SAMPLE_DIR, injuries_as_of, parse_date
from agent.teams import normalize_abbr

# Rolling window, in games. Ten is what `team_form_as_of` uses and what the
# agent's team_form tool reports, so the model and the agent describe form the
# same way. Changing it here without changing it there breaks the equivalence
# test, which is the intended alarm.
FORM_WINDOW = 10

# Order is the contract. `models/win_probability.json` stores coefficients
# positionally, so appending is safe and reordering silently retrains meaning
# onto the wrong weight. Append only.
FEATURE_NAMES = (
    "form_margin_diff",
    "win_pct_diff",
    "rest_diff",
    "home_back_to_back",
    "away_back_to_back",
    "injury_weight_diff",
    "home_games_played",
    "away_games_played",
)


@dataclass(frozen=True)
class GameRow:
    """One training or scoring example."""

    game_id: str
    game_date: date
    home: str
    away: str
    features: tuple[float, ...]
    home_won: int | None  # None when the game has not been played
    playoffs: bool


class _TeamState:
    """Everything we are allowed to know about one team, at one instant."""

    __slots__ = ("margins", "wins", "played", "last_game_date")

    def __init__(self) -> None:
        self.margins: deque[int] = deque(maxlen=FORM_WINDOW)
        self.wins = 0
        self.played = 0
        self.last_game_date: date | None = None

    @property
    def form_margin(self) -> float:
        return sum(self.margins) / len(self.margins) if self.margins else 0.0

    @property
    def win_pct(self) -> float:
        # 0.5 rather than 0.0 for a team with no games: an unplayed team is
        # unknown, not terrible, and 0.0 would hand the opponent a fake edge in
        # every opening-week row.
        return self.wins / self.played if self.played else 0.5

    def rest_days(self, on: date) -> int | None:
        if self.last_game_date is None:
            return None
        return (on - self.last_game_date).days - 1


def _season_files() -> dict[int, str]:
    return {
        int(p.stem.rsplit("_", 1)[1]): str(p)
        for p in sorted(SAMPLE_DIR.glob("game_logs_*.csv"))
    }


def _injury_weight(abbr: str, as_of: date, cache: dict) -> float:
    """Total importance of the players a team is missing.

    Cached per (team, date): the replay walks the whole injury log, and the
    same team-date pair recurs constantly across a season.
    """
    key = (abbr, as_of)
    if key not in cache:
        try:
            out = injuries_as_of(abbr, as_of)
            cache[key] = sum(i.get("importance") or 0.0 for i in out)
        except Exception:
            # An unreadable injury log must not silently become "nobody hurt" in
            # one season and a real number in another -- that would make the
            # feature mean different things across the train/test boundary. 0.0
            # is still the value, but the caller sees the log's coverage window
            # in build_season's report.
            cache[key] = 0.0
    return cache[key]


def _replay_injury_cutoff(game_date: date) -> date:
    """Last safe day for a date-only injury source in a historical replay."""
    return game_date - timedelta(days=1)


@lru_cache(maxsize=8)
def build_season(
    season: int, *, with_injuries: bool = True
) -> tuple[list[GameRow], dict]:
    """Walk one season forward, emitting a feature row per game before scoring it.

    Results are cached by season and injury mode because the historical files
    are immutable during a UI/evaluation process. Returns the rows plus a small
    report, so a caller can tell the difference
    between "no injuries that day" and "the injury log does not cover this
    season" without reading the log itself.
    """
    files = _season_files()
    if season not in files:
        raise FileNotFoundError(
            f"no game_logs_{season}.csv in {SAMPLE_DIR}. "
            "Run scripts/fetch_game_logs.py."
        )

    with open(files[season], newline="", encoding="utf-8") as fh:
        games = list(csv.DictReader(fh))
    games.sort(key=lambda g: (g["game_date"], g["game_id"]))

    state: dict[str, _TeamState] = defaultdict(_TeamState)
    injury_cache: dict = {}
    rows: list[GameRow] = []
    injury_hits = 0

    for g in games:
        gd = parse_date(g["game_date"])
        home, away = normalize_abbr(g["home"]), normalize_abbr(g["away"])
        hs, as_ = state[home], state[away]

        h_rest, a_rest = hs.rest_days(gd), as_.rest_days(gd)
        # No prior game means no rest number. Treating that as 0 would label
        # every season opener a back-to-back; 2 is the league's typical gap and
        # keeps the opener neutral instead of alarming.
        h_rest_f = 2.0 if h_rest is None else float(h_rest)
        a_rest_f = 2.0 if a_rest is None else float(a_rest)

        if with_injuries:
            # The historical injury source records a calendar date, not a
            # publication timestamp.  A transaction dated on game day might
            # have been posted after tip-off, so it is not safe in a replay.
            # Stop at the previous calendar day unless/until the source is
            # replaced by timestamped official injury reports.
            injury_cutoff = _replay_injury_cutoff(gd)
            h_inj = _injury_weight(home, injury_cutoff, injury_cache)
            a_inj = _injury_weight(away, injury_cutoff, injury_cache)
            if h_inj or a_inj:
                injury_hits += 1
        else:
            h_inj = a_inj = 0.0

        features = (
            hs.form_margin - as_.form_margin,
            hs.win_pct - as_.win_pct,
            h_rest_f - a_rest_f,
            1.0 if h_rest == 0 else 0.0,
            1.0 if a_rest == 0 else 0.0,
            h_inj - a_inj,
            float(hs.played),
            float(as_.played),
        )

        home_pts, away_pts = g.get("home_pts"), g.get("away_pts")
        played = bool(home_pts) and bool(away_pts)
        home_won = int(int(home_pts) > int(away_pts)) if played else None

        rows.append(
            GameRow(
                game_id=g["game_id"],
                game_date=gd,
                home=home,
                away=away,
                features=features,
                home_won=home_won,
                playoffs=g.get("playoffs") == "1",
            )
        )

        # ---- only now may this game exist. Everything above saw the past only.
        if played:
            margin = int(home_pts) - int(away_pts)
            hs.margins.append(margin)
            as_.margins.append(-margin)
            hs.wins += margin > 0
            as_.wins += margin < 0
            hs.played += 1
            as_.played += 1
        hs.last_game_date = gd
        as_.last_game_date = gd

    return rows, {
        "season": season,
        "games": len(rows),
        "games_with_injury_signal": injury_hits,
        "injuries_used": with_injuries,
    }


def live_features(
    home_abbr: str,
    away_abbr: str,
    as_of: date,
    game_date: date | None = None,
) -> tuple[float, ...]:
    """Features for a game that is not in the logs yet -- the UI's path.

    Replays the season up to as_of and reads the two teams' accumulators. Slower
    than build_season per call, and that is the right trade: this runs once for
    one game a user asked about, not 3,777 times in a training loop.
    """
    from agent.sources import season_end_year

    home, away = normalize_abbr(home_abbr), normalize_abbr(away_abbr)
    target_date = game_date or (as_of + timedelta(days=1))
    if as_of >= target_date:
        raise ValueError("as_of must be before game_date")
    season = season_end_year(target_date)

    try:
        rows, _ = build_season(season)
    except FileNotFoundError:
        rows = []

    state: dict[str, _TeamState] = defaultdict(_TeamState)
    scheduled_last: dict[str, date] = {}
    for r in rows:
        if r.game_date < target_date:
            scheduled_last[r.home] = r.game_date
            scheduled_last[r.away] = r.game_date
        if r.game_date <= as_of and r.home_won is not None:
            hs, as_s = state[r.home], state[r.away]
            hs.played += 1
            as_s.played += 1
            hs.wins += r.home_won == 1
            as_s.wins += r.home_won == 0

    hs, as_s = state[home], state[away]
    h_last, a_last = scheduled_last.get(home), scheduled_last.get(away)
    h_rest = (target_date - h_last).days - 1 if h_last else None
    a_rest = (target_date - a_last).days - 1 if a_last else None
    h_rest_f = 2.0 if h_rest is None else float(h_rest)
    a_rest_f = 2.0 if a_rest is None else float(a_rest)

    cache: dict = {}
    h_inj = _injury_weight(home, as_of, cache)
    a_inj = _injury_weight(away, as_of, cache)

    # Margins come from the gated accessor the agent uses, so a one-off UI
    # prediction and the agent's team_form tool cannot disagree on form.
    from agent.sources import team_form_as_of

    hf = team_form_as_of(home, as_of, FORM_WINDOW)
    af = team_form_as_of(away, as_of, FORM_WINDOW)
    h_margin = hf["avg_point_diff"] if hf else 0.0
    a_margin = af["avg_point_diff"] if af else 0.0

    return (
        h_margin - a_margin,
        hs.win_pct - as_s.win_pct,
        h_rest_f - a_rest_f,
        1.0 if h_rest == 0 else 0.0,
        1.0 if a_rest == 0 else 0.0,
        h_inj - a_inj,
        float(hs.played),
        float(as_s.played),
    )
