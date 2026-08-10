"""Data sources behind the agent's tools.

Two implementations share one interface:

  MockSource  fixed JSON fixture, deterministic, no data files needed.
  CsvSource   the real datasets on main, filtered so nothing published after
              as_of_date can reach the agent.

Leakage rules enforced here (the 2026-07-07 class decision):

1. Injuries come from a transaction log. We replay it forward and stop at
   as_of_date, so the injury list is what a person could have known that
   morning -- not the season-long summary.

2. Team ratings and player averages in nba_stats are END-OF-SEASON aggregates.
   A 2024-25 rating row already contains games played after any mid-season
   as_of_date, so using the current season would leak. We therefore serve the
   PRIOR completed season and label it. Current-season, as-of-accurate ratings
   need game logs (see rest/h2h below).

3. Rest, back-to-back, and head-to-head need a game-by-game schedule. That
   dataset does not exist on main yet. When it is missing we return nulls with
   a reason -- we never guess a number.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from agent.teams import abbr_from_nickname, full_name, normalize_abbr, odds_abbr

REPO_ROOT = Path(__file__).resolve().parents[1]

# Set NBA_SNAPSHOT_DIR to a directory built by scripts/gate_snapshot.py and the
# whole data layer reads from there instead -- a copy that physically never
# held anything after the cutoff. Unset (the default) reads the full data
# directory. Either way the query-time filters below still apply: the snapshot
# removes what nobody may see, the filters decide what each tool may see.
DATA_ROOT = Path(os.environ.get("NBA_SNAPSHOT_DIR") or REPO_ROOT / "data")
MOCK_DIR = REPO_ROOT / "data" / "mock"
RAW_DIR = DATA_ROOT / "raw"
SAMPLE_DIR = DATA_ROOT / "samples"

INJURY_CSV = RAW_DIR / "injury_data_2016_2025" / "injury_data.csv"
# The Kaggle set stops at 2025-01-12, which left the 2025-26 replay window with
# no injuries at all. The second file continues the same log from 2025-01-13
# (scraped from the same upstream source, identical columns) so the two join
# without a gap or an overlap. Kept as separate files to preserve provenance.
INJURY_CSVS = (
    INJURY_CSV,
    RAW_DIR / "injury_pst_2025_2026" / "injury_data.csv",
)
TEAM_SUMMARY_CSV = RAW_DIR / "nba_stats_1947_present" / "Team Summaries.csv"
PLAYER_PER_GAME_CSV = RAW_DIR / "nba_stats_1947_present" / "Player Per Game.csv"
ODDS_CSV = SAMPLE_DIR / "odds_only.csv"
# Features only -- no points, rebounds, assists or outcome. Built by
# `python -m models.stat_line_features` from Patrick's engineered export, which
# keeps the box-score result in the same row as the features. Same hazard as the
# odds file keeping score_home beside the line, handled the same way.
PLAYER_FEATURES_CSV = SAMPLE_DIR / "player_features_2026.csv"
PLAYER_STATS_CSV = DATA_ROOT / "exports" / "player_stats_engineered.csv"


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def season_end_year(on: date) -> int:
    """NBA seasons span two calendar years; nba_stats labels them by end year.

    A game on 2024-12-20 belongs to the 2024-25 season, labelled 2025.
    """
    return on.year + 1 if on.month >= 8 else on.year


def parse_matchup_id(matchup_id: str) -> tuple[str, str, date]:
    """'LAL-BOS-2026-01-15' -> (away 'LAL', home 'BOS', 2026-01-15)."""
    parts = matchup_id.strip().split("-")
    if len(parts) != 5:
        raise ValueError(
            f"matchup_id must look like AWAY-HOME-YYYY-MM-DD, got {matchup_id!r}"
        )
    away, home, y, m, d = parts
    return normalize_abbr(away), normalize_abbr(home), parse_date(f"{y}-{m}-{d}")


# --------------------------------------------------------------------------
# Real data readers (cached: these files are read once per process)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _injury_rows() -> tuple[tuple[date, str, str, str, str], ...]:
    """(date, team_abbr, player, direction, note), sorted by date.

    direction is 'out' (Relinquished) or 'back' (Acquired).
    """
    rows: list[tuple[date, str, str, str, str]] = []
    for path in INJURY_CSVS:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                abbr = abbr_from_nickname(r["Team"])
                if not abbr:
                    continue
                note = r["Notes"].strip()
                acquired = r["Acquired"].strip()
                relinquished = r["Relinquished"].strip()
                if relinquished:
                    rows.append(
                        (parse_date(r["Date"]), abbr, relinquished, "out", note)
                    )
                elif acquired:
                    rows.append((parse_date(r["Date"]), abbr, acquired, "back", note))
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


def injury_data_through() -> date | None:
    rows = _injury_rows()
    return rows[-1][0] if rows else None


# A player relinquished this long ago with no return record has almost
# certainly left the team (the log records IL moves, never departures), so
# treating them as "still injured" is wrong. One NBA season is ~8 months; a
# genuine season-ending injury resolves inside that.
STALE_INJURY_DAYS = 240


def injuries_as_of(team_abbr: str, as_of: date) -> list[dict]:
    """Replay the transaction log to as_of; return who is still out.

    This is the date-gated primitive: a player is out if they were relinquished
    on or before as_of and have not been re-acquired since. Nothing dated after
    as_of is read.

    Two corrections the raw log needs, both computed only from records <= as_of:

    1. The log has no "traded away" event. A player who goes on IL and is then
       relinquished/acquired by ANOTHER team has left; without this check they
       stay on the old team's out-list forever (Kemba Walker was still "out"
       for Boston in 2024).
    2. A relinquish older than STALE_INJURY_DAYS with no return is a departure
       the log never recorded, not an active injury. Drop it and count it.
    """
    team_abbr = normalize_abbr(team_abbr)
    out: dict[str, dict] = {}
    last_team: dict[str, str] = {}

    for when, abbr, player, direction, note in _injury_rows():
        if when > as_of:
            break
        last_team[player] = abbr
        if abbr != team_abbr:
            continue
        if direction == "out":
            out[player] = {
                "team": team_abbr,
                "player": player,
                "status": "Out",
                "note": note,
                "published": when.isoformat(),
            }
        else:
            out.pop(player, None)

    active = []
    for player, inj in out.items():
        if last_team.get(player) != team_abbr:
            continue  # showed up on another team since -- no longer ours
        age = (as_of - parse_date(inj["published"])).days
        if age > STALE_INJURY_DAYS:
            continue  # unrecorded departure, not an injury
        inj["days_out"] = age
        inj.update(player_importance(player, as_of))
        active.append(inj)

    # importance is None for players with no prior season, so `or 0.0` not a default
    return sorted(active, key=lambda i: -(i.get("importance") or 0.0))


def player_is_out(player_name: str, team_abbrs: tuple[str, ...], as_of: date) -> bool:
    """Whether the gated injury log lists a matchup player as out.

    Some transaction rows join multiple aliases with a slash. Keep that raw-data
    detail inside the source boundary so every prediction path applies the same
    name matching and cutoff.
    """
    requested = player_name.strip().casefold()
    for team_abbr in team_abbrs:
        for injury in injuries_as_of(team_abbr, as_of):
            aliases = {
                alias.strip().casefold()
                for alias in str(injury.get("player", "")).split("/")
            }
            if requested in aliases:
                return True
    return False


def player_importance(player_name: str, as_of: date) -> dict:
    """How much this player actually matters, from the PRIOR completed season.

    The advisor's 2026-07-21 note: the injury list weighted a franchise player
    and a tenth man identically, so a team losing its best player looked the
    same as a team losing a bench body.

    Minutes carry most of the signal -- a coach's own revealed ranking of who
    matters -- with scoring as a secondary term. Prior season only, same gating
    rule as team ratings: using the current season mid-year would leak.
    """
    row = player_season_averages(player_name, season_end_year(as_of) - 1)
    if not row:
        return {
            "importance": None,
            "tier": "unknown",
            "importance_basis": "no prior season",
        }

    minutes = min((row.get("min_avg") or 0.0) / 36.0, 1.0)
    points = min((row.get("pts_avg") or 0.0) / 28.0, 1.0)
    score = round(0.6 * minutes + 0.4 * points, 3)

    if score >= 0.70:
        tier = "star"
    elif score >= 0.45:
        tier = "starter"
    elif score >= 0.25:
        tier = "rotation"
    else:
        tier = "bench"

    return {
        "importance": score,
        "tier": tier,
        "importance_basis": row.get("basis", "prior completed season"),
    }


@lru_cache(maxsize=1)
def _team_summaries() -> dict[tuple[int, str], dict]:
    if not TEAM_SUMMARY_CSV.exists():
        return {}
    table: dict[tuple[int, str], dict] = {}
    with TEAM_SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["lg"] != "NBA" or not r["abbreviation"] or not r["season"]:
                continue
            table[(int(r["season"]), normalize_abbr(r["abbreviation"]))] = r
    return table


def team_ratings(abbr: str, season: int) -> dict | None:
    row = _team_summaries().get((season, normalize_abbr(abbr)))
    if not row:
        return None

    def num(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    return {
        "abbr": normalize_abbr(abbr),
        "name": full_name(abbr),
        "record": f"{row['w']}-{row['l']}",
        "off_rating": num("o_rtg"),
        "def_rating": num("d_rtg"),
        "pace": num("pace"),
        "basis": f"{season - 1}-{str(season)[2:]} final (prior completed season)",
    }


@lru_cache(maxsize=1)
def _player_per_game() -> dict[tuple[int, str], dict]:
    if not PLAYER_PER_GAME_CSV.exists():
        return {}
    table: dict[tuple[int, str], dict] = {}
    with PLAYER_PER_GAME_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["lg"] != "NBA" or not r["season"]:
                continue
            key = (int(r["season"]), r["player"].strip().lower())
            # A traded player has one row per team plus a combined row; the
            # combined row lists the most games, so keep the max.
            prev = table.get(key)
            if prev is None or int(r["g"] or 0) > int(prev["g"] or 0):
                table[key] = r
    return table


def player_season_averages(player_name: str, season: int) -> dict | None:
    row = _player_per_game().get((season, player_name.strip().lower()))
    if not row:
        return None

    def num(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    return {
        "name": row["player"],
        "team": normalize_abbr(row["team"]),
        "games": int(row["g"] or 0),
        "pts_avg": num("pts_per_game"),
        "reb_avg": num("trb_per_game"),
        "ast_avg": num("ast_per_game"),
        "min_avg": num("mp_per_game"),
        "basis": f"{season - 1}-{str(season)[2:]} final (prior completed season)",
    }


def _game_log_path(season: int) -> Path:
    return SAMPLE_DIR / f"game_logs_{season}.csv"


@lru_cache(maxsize=4)
def _game_logs(season: int) -> tuple[dict, ...]:
    path = _game_log_path(season)
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def _all_game_logs() -> tuple[dict, ...]:
    """Every season we have on disk. Rest is a within-season question, but
    head-to-head history reaches back across seasons."""
    games: list[dict] = []
    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.glob("game_logs_*.csv")):
            with path.open(newline="", encoding="utf-8") as f:
                games.extend(csv.DictReader(f))
    return tuple(games)


NO_SCHEDULE = (
    "No game-log dataset on main for this season, so rest/back-to-back/H2H "
    "cannot be computed. Run scripts/fetch_game_logs.py, or use --source mock. "
    "This is the open data-layer gap (schedule + results)."
)


def schedule_context(away: str, home: str, game_date: date, as_of: date) -> dict:
    """Rest, back-to-back, and H2H from game logs -- strictly before as_of.

    Returns nulls plus a reason when the schedule dataset is absent. We would
    rather show the agent a null than a number we made up.
    """
    season = season_end_year(game_date)
    logs = _game_logs(season)
    if not logs:
        return {
            "rest": {
                "home_days_rest": None,
                "away_days_rest": None,
                "away_back_to_back": None,
                "home_back_to_back": None,
                "unavailable": NO_SCHEDULE,
            },
            "h2h_last_5": [],
            "h2h_unavailable": NO_SCHEDULE,
        }

    def last_game_before(abbr: str) -> date | None:
        # Rest uses SCHEDULE DATES, not results. The NBA publishes the full
        # schedule in August, so "BOS plays Dec 23 and Dec 25" is knowable on
        # any as_of date -- it is not leakage. Only outcomes are gated (see
        # h2h below). Gating dates at as_of would wrongly report 53 days rest
        # for a game scouted seven weeks out.
        played = [
            parse_date(g["game_date"])
            for g in logs
            if parse_date(g["game_date"]) < game_date
            and normalize_abbr(abbr)
            in (normalize_abbr(g["home"]), normalize_abbr(g["away"]))
        ]
        return max(played) if played else None

    def days_rest(abbr: str) -> int | None:
        last = last_game_before(abbr)
        return (game_date - last).days - 1 if last else None

    home_rest, away_rest = days_rest(home), days_rest(away)

    # Rest is a within-season question; head-to-head reaches back across every
    # season on disk. Both stay strictly at or before as_of.
    h2h = [
        {
            "date": g["game_date"],
            "winner": normalize_abbr(g["winner"]),
            "score": f"{g['home_pts']}-{g['away_pts']}",
            "home": normalize_abbr(g["home"]),
        }
        for g in _all_game_logs()
        if parse_date(g["game_date"]) <= as_of
        and {normalize_abbr(g["home"]), normalize_abbr(g["away"])}
        == {normalize_abbr(home), normalize_abbr(away)}
    ]
    h2h.sort(key=lambda g: g["date"], reverse=True)

    return {
        "rest": {
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "away_back_to_back": away_rest == 0 if away_rest is not None else None,
            "home_back_to_back": home_rest == 0 if home_rest is not None else None,
        },
        "h2h_last_5": h2h[:5],
    }


from datetime import timedelta  # noqa: E402

_ONE_DAY = timedelta(days=1)


def team_form_as_of(team_abbr: str, as_of: date, last_n: int = 10) -> dict | None:
    """A team's CURRENT strength from games PLAYED before as_of, not last season.

    The stale-ratings fix. Team ratings on file are end-of-season aggregates, so
    mid-season they either leak (current season) or go stale (prior season, wrong
    by December). This reads the game logs -- outcomes gated at as_of -- and
    builds a rolling record and average point differential, a light net-rating
    proxy that actually reflects who the team is right now.

    Results are gated: a game counts only once it has been played AND is on or
    before as_of. Returns None when there are no in-season games yet (opening
    week), so the caller falls back to prior-season ratings rather than guess.
    """
    team = normalize_abbr(team_abbr)
    season = season_end_year(as_of)
    played = []
    for g in _game_logs(season):
        gd = parse_date(g["game_date"])
        if gd >= as_of:
            continue  # not yet played, or the game itself -- would leak
        h, a = normalize_abbr(g["home"]), normalize_abbr(g["away"])
        if team not in (h, a):
            continue
        is_home = team == h
        pf = int(g["home_pts"] if is_home else g["away_pts"])
        pa = int(g["away_pts"] if is_home else g["home_pts"])
        played.append((gd, pf - pa, pf > pa))

    if not played:
        return None

    played.sort(key=lambda x: x[0])
    window = played[-last_n:]
    wins = sum(1 for _, _, w in played if w)
    recent_wins = sum(1 for _, _, w in window if w)
    avg_margin = sum(m for _, m, _ in window) / len(window)

    return {
        "abbr": team,
        "as_of": as_of.isoformat(),
        "games_played": len(played),
        "record": f"{wins}-{len(played) - wins}",
        "last_n": len(window),
        "recent_record": f"{recent_wins}-{len(window) - recent_wins}",
        "avg_point_diff": round(avg_margin, 2),
        "basis": f"rolling over last {len(window)} games before {as_of.isoformat()}",
    }


# --------------------------------------------------------------------------
# Historical betting lines
# --------------------------------------------------------------------------

# The file stores one row per game: the CLOSING line, the market's final
# number. That is a fact about tip-off, not about the morning of. It is the
# benchmark we score a prediction against -- it is not an as-of feature, and
# feeding it to the model would teach the model to copy Vegas.
CLOSING_LINE_CAVEAT = (
    "Closing line -- the market's final number, not knowable until tip-off. "
    "Score predictions against it; never hand it to a model as an as-of feature."
)

# moneyline_away/moneyline_home are empty from the 2023-24 season onward,
# which is every season we actually test on. Spread and total are populated
# throughout. Say so rather than returning a silent null.
NO_MONEYLINE = (
    "moneyline: the source file carries none from the 2023-24 season onward, "
    "which includes the entire 2025-26 replay window. Spread and total are present."
)


@lru_cache(maxsize=1)
def _odds_rows() -> dict[tuple[str, str, str], dict]:
    """(away, home, iso date) -> row, keyed in the odds file's own spellings.

    Callers come in with repo abbreviations and go through teams.odds_abbr.
    """
    if not ODDS_CSV.exists():
        return {}
    table: dict[tuple[str, str, str], dict] = {}
    with ODDS_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (
                r["away"].strip().lower(),
                r["home"].strip().lower(),
                r["date"].strip(),
            )
            table[key] = r
    return table


def closing_line(away: str, home: str, game_date: date, as_of: date) -> dict:
    """The closing line for one game, or a stated reason there is none.

    Gated the same way matchup_context is: a query dated after tip-off is
    refused. The line sits one column away from the result in the source data,
    so it is exactly the field a replay must not be able to reach backwards for.
    """
    if as_of > game_date:
        return {
            "status": "gated",
            "reason": (
                f"as_of_date {as_of.isoformat()} is after tip-off "
                f"{game_date.isoformat()}. Withheld: a replay query must not be "
                "able to reach past the game it is predicting."
            ),
        }

    row = _odds_rows().get((odds_abbr(away), odds_abbr(home), game_date.isoformat()))
    if row is None:
        return {
            "status": "not_found",
            "reason": (
                f"No odds row for {normalize_abbr(away)} at {normalize_abbr(home)} "
                f"on {game_date.isoformat()}."
            ),
        }

    def num(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    spread = num("spread")
    favored = row["whos_favored"].strip().lower()
    favorite = normalize_abbr(home if favored == "home" else away) if favored else None

    # The file stores an unsigned magnitude plus whos_favored, so "spread 6.5"
    # alone cannot say who is laying the points. Sign it per team: the favorite
    # is negative, the underdog positive.
    spread_home = spread_away = None
    if spread is not None and favored in ("home", "away"):
        home_favored = favored == "home"
        spread_home = -spread if home_favored else spread
        spread_away = spread if home_favored else -spread

    ml_away, ml_home = num("moneyline_away"), num("moneyline_home")

    return {
        "status": "ok",
        "line_type": "closing",
        "favorite": favorite,
        "spread": spread,
        "spread_home": spread_home,
        "spread_away": spread_away,
        "total": num("total"),
        "moneyline_away": ml_away,
        "moneyline_home": ml_home,
        "unavailable": [] if ml_home is not None else [NO_MONEYLINE],
        "caveat": CLOSING_LINE_CAVEAT,
    }


def slate_as_of(as_of: date, days_ahead: int = 1) -> dict:
    """The fixtures tipping off in the days after as_of.

    WHY THIS IS NOT LEAKAGE. The NBA publishes its schedule in August, so who plays
    whom on a future date is knowable on any as_of_date. The RESULT is not, and the
    game-log rows this reads carry `home_pts`, `away_pts` and `winner` right next to
    the fixture. Only the four identity fields are copied out; the score columns are
    never touched, and `tests/test_date_gating.py` asserts they never appear.
    """
    first = as_of + timedelta(days=1)
    last = as_of + timedelta(days=max(1, days_ahead))
    games = []
    for row in _all_game_logs():
        played = parse_date(row["game_date"])
        if first <= played <= last:
            games.append(
                {
                    "matchup_id": row["game_id"],
                    "date": row["game_date"],
                    "away": normalize_abbr(row["away"]),
                    "home": normalize_abbr(row["home"]),
                }
            )
    games.sort(key=lambda g: (g["date"], g["matchup_id"]))
    return {
        "source": "real",
        "as_of_date": as_of.isoformat(),
        "window": {"from": first.isoformat(), "to": last.isoformat()},
        "games": games,
        "count": len(games),
        "caveat": (
            "Fixtures only: teams and dates, taken from the season's game log. Tip-off "
            "times are not in that dataset, so this cannot tell you when a game starts. "
            "No score or result is read."
        ),
    }


@lru_cache(maxsize=1)
def _player_feature_rows() -> tuple[dict, ...]:
    if not PLAYER_STATS_CSV.exists():
        return ()
    with PLAYER_STATS_CSV.open(encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


def player_features_as_of(
    player_name: str,
    away: str,
    home: str,
    game_date: date,
    as_of: date,
) -> dict:
    """One player's as-of form for a specific game, gated at as_of.

    THE GATE, precisely. Feature rows are box-score-derived, so the existence of a
    row after ``as_of`` would itself reveal that the player appeared in a future
    game. We therefore select only rows dated on or before ``as_of``. The latest
    row matching the target venue supplies an observable historical snapshot; the
    published team schedule supplies target-game rest and back-to-back context.
    """
    if as_of >= game_date:
        raise ValueError(
            f"as_of_date {as_of.isoformat()} is not before tip-off "
            f"{game_date.isoformat()}: the trailing averages for that game would "
            "then include games played after as_of."
        )

    wanted = player_name.strip().casefold()
    observed = [
        r
        for r in _player_feature_rows()
        if r.get("name", "").strip().casefold() == wanted
        and parse_date(r["game_date"]) <= as_of
    ]
    observed.sort(key=lambda r: r["game_date"])
    if not observed:
        return {
            "available": False,
            "reason": "No observable pregame feature snapshot is available.",
        }

    latest = max(observed, key=lambda r: r["game_date"])
    home_key = full_name(home).upper().replace(" ", "_")
    away_key = full_name(away).upper().replace(" ", "_")
    player_team = latest.get("team", "").strip().upper().replace(" ", "_")
    if player_team == home_key:
        is_home, player_abbr, opponent = 1, home, away_key
    elif player_team == away_key:
        is_home, player_abbr, opponent = 0, away, home_key
    else:
        return {
            "available": False,
            "reason": (
                f"{player_name}'s latest observable team is not part of this matchup."
            ),
        }

    target_location = "HOME" if is_home else "AWAY"
    venue_rows = [
        r
        for r in observed
        if r.get("location", "").strip().upper() == target_location
    ]
    if not venue_rows:
        return {
            "available": False,
            "reason": (
                f"No {target_location.lower()} feature snapshot for {player_name} "
                f"is available on or before {as_of.isoformat()}."
            ),
        }
    row = latest

    def mean(rows: list[dict], key: str) -> float | None:
        values = []
        for item in rows:
            try:
                values.append(float(item[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return sum(values) / len(values) if values else None

    features: dict[str, float | None] = {}
    for window in (5, 10):
        recent = observed[-window:]
        for source, stem in [
            ("points", "pts"),
            ("total_rebounds", "reb"),
            ("assists", "ast"),
            ("minutes", "min"),
        ]:
            features[f"rolling_{stem}_{window}"] = mean(recent, source)
        if window != 5:
            continue
        for made, attempted, output in [
            ("made_field_goals", "attempted_field_goals", f"rolling_fg_pct_{window}"),
            (
                "made_three_point_field_goals",
                "attempted_three_point_field_goals",
                f"rolling_3p_pct_{window}",
            ),
        ]:
            made_sum = sum(float(r.get(made) or 0) for r in recent)
            attempted_sum = sum(float(r.get(attempted) or 0) for r in recent)
            features[output] = made_sum / attempted_sum if attempted_sum else None
    features["home_away_pts_avg"] = mean(venue_rows, "points")
    features["home_away_reb_avg"] = mean(venue_rows, "total_rebounds")
    features["home_away_ast_avg"] = mean(venue_rows, "assists")
    scheduled_dates = [
        parse_date(g["game_date"])
        for g in _game_logs(season_end_year(game_date))
        if parse_date(g["game_date"]) < game_date
        and player_abbr
        in {normalize_abbr(g["home"]), normalize_abbr(g["away"])}
    ]
    if scheduled_dates:
        rest_days = (game_date - max(scheduled_dates)).days
        features["rest_days"] = float(rest_days)
        features["is_back_to_back"] = 1.0 if rest_days == 1 else 0.0
    features["is_home"] = float(is_home)

    return {
        "available": True,
        "player": row.get("name"),
        "team": player_team,
        "opponent": opponent,
        "game_date": game_date.isoformat(),
        "feature_snapshot_date": row["game_date"],
        "features": features,
    }


# Kept here rather than imported from models/ so the data layer does not depend on
# the model layer; models/train_stat_line.py asserts the two lists agree.
STAT_LINE_FEATURE_KEYS = (
    "rolling_pts_5",
    "rolling_reb_5",
    "rolling_ast_5",
    "rolling_min_5",
    "rolling_pts_10",
    "rolling_reb_10",
    "rolling_ast_10",
    "rolling_min_10",
    "rolling_fg_pct_5",
    "rolling_3p_pct_5",
    "home_away_pts_avg",
    "home_away_reb_avg",
    "home_away_ast_avg",
    "rest_days",
    "is_back_to_back",
    "is_home",
)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


class MockSource:
    """Deterministic fixture. No data files, no network, safe for tests."""

    name = "mock"

    def _fixture(self, matchup_id: str | None = None) -> dict:
        path = MOCK_DIR / "matchup_lal_bos.json"
        if matchup_id:
            candidate = (
                MOCK_DIR / f"matchup_{matchup_id.lower().replace('-', '_')}.json"
            )
            if candidate.exists():
                path = candidate
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def matchup_context(self, matchup_id: str, as_of_date: str) -> dict:
        data = self._fixture(matchup_id)
        return {
            "source": self.name,
            "as_of_date": as_of_date,
            "matchup_id": data["matchup_id"],
            "game_date": data["game_date"],
            "home_team": data["home_team"],
            "away_team": data["away_team"],
            "rest": data["rest"],
            "injuries": [i for i in data["injuries"] if i["published"] <= as_of_date],
            "h2h_last_5": [g for g in data["h2h_last_5"] if g["date"] <= as_of_date],
        }

    def injuries(self, team_abbr: str, as_of_date: str) -> dict:
        """Who was known to be out, as of a date. Mock fixture."""
        ctx = self.matchup_context("LAL-BOS-2026-01-15", as_of_date)
        out = [i for i in ctx["injuries"] if i["team"] == team_abbr.upper()]
        return {
            "source": "mock",
            "team": team_abbr.upper(),
            "as_of_date": as_of_date,
            "injuries": out,
        }

    def player_splits(self, player_name: str, back_to_back: bool = False) -> dict:
        data = self._fixture()
        for p in data["key_players"]:
            if p["name"].lower() == player_name.lower():
                out = {
                    "source": self.name,
                    "name": p["name"],
                    "team": p["team"],
                    "pts_avg": p["pts_avg"],
                    "reb_avg": p["reb_avg"],
                    "ast_avg": p["ast_avg"],
                }
                if back_to_back:
                    out["b2b_pts_avg"] = p["b2b_pts_avg"]
                    out["note"] = (
                        "Player is on a back-to-back; prefer b2b split over season avg."
                    )
                return out
        return {"error": f"player not found: {player_name}"}

    def schedule(self, as_of_date: str, days_ahead: int = 1) -> dict:
        """The fixture list is real data, not a fixture value, so mock says so."""
        return {
            "source": "mock",
            "as_of_date": as_of_date,
            "games": [],
            "count": 0,
            "caveat": "MockSource carries no schedule; run with --source real.",
        }

    def player_features(
        self, player_name: str, matchup_id: str, as_of_date: str
    ) -> dict:
        # The fixture has no game-by-game history, so there are no trailing
        # averages to serve. Gate first anyway, so mock and real refuse an
        # after-tip query identically.
        _, _, game_date = parse_matchup_id(matchup_id)
        as_of = parse_date(as_of_date)
        if as_of >= game_date:
            raise ValueError(
                f"as_of_date {as_of_date} is not before tip-off "
                f"{game_date.isoformat()}."
            )
        return {
            "available": False,
            "reason": "MockSource carries no game logs; run with --source real.",
        }

    def betting_line(self, matchup_id: str, as_of_date: str) -> dict:
        # The fixture carries no odds. Gate first anyway, so mock and real
        # refuse an after-tip query identically.
        _, _, game_date = parse_matchup_id(matchup_id)
        if parse_date(as_of_date) > game_date:
            return {
                "status": "gated",
                "reason": (
                    f"as_of_date {as_of_date} is after tip-off {game_date.isoformat()}."
                ),
            }
        return {
            "status": "not_found",
            "reason": "The mock fixture carries no betting line. Use --source real.",
        }


class CsvSource:
    """The real datasets on main, date-gated at as_of."""

    name = "real"

    def matchup_context(self, matchup_id: str, as_of_date: str) -> dict:
        away, home, game_date = parse_matchup_id(matchup_id)
        as_of = parse_date(as_of_date)
        if as_of > game_date:
            raise ValueError(
                f"as_of_date {as_of_date} is after tip-off {game_date.isoformat()}: "
                "that would leak the result the agent is meant to predict."
            )

        season = season_end_year(game_date)
        prior = season - 1  # leakage-safe: a completed season before as_of

        warnings: list[str] = []
        through = injury_data_through()
        if through and as_of > through:
            warnings.append(
                f"Injury log ends {through.isoformat()}, before as_of {as_of_date}. "
                "Injuries shown are stale, not current."
            )

        home_ratings = team_ratings(home, prior)
        away_ratings = team_ratings(away, prior)
        for abbr, ratings in ((home, home_ratings), (away, away_ratings)):
            if ratings is None:
                warnings.append(f"No {prior} season ratings for {abbr}.")

        payload = {
            "source": self.name,
            "as_of_date": as_of_date,
            "matchup_id": matchup_id,
            "game_date": game_date.isoformat(),
            "home_team": home_ratings or {"abbr": home, "name": full_name(home)},
            "away_team": away_ratings or {"abbr": away, "name": full_name(away)},
            "injuries": injuries_as_of(home, as_of) + injuries_as_of(away, as_of),
            "ratings_basis": (
                f"Team ratings are {prior - 1}-{str(prior)[2:]} final. Current-season "
                "as-of ratings would require game logs (not on main yet); using the "
                "in-progress season's final numbers would leak post-as_of games."
            ),
        }
        payload.update(schedule_context(away, home, game_date, as_of))
        if warnings:
            payload["warnings"] = warnings
        return payload

    def injuries(self, team_abbr: str, as_of_date: str) -> dict:
        """Who was known to be out that morning -- the log replayed, stopped at as_of.

        Each entry carries an `importance` (0-1) and a `tier`, so a 10th man and a
        franchise player no longer weigh the same. The remaining limit is stated in
        the payload rather than hidden: the log ends 2025-01-12.
        """
        as_of = parse_date(as_of_date)
        payload = {
            "source": "real",
            "team": normalize_abbr(team_abbr),
            "as_of_date": as_of_date,
            "injuries": injuries_as_of(normalize_abbr(team_abbr), as_of),
            "importance_basis": (
                "importance = 0.6*(min/36) + 0.4*(pts/28) from the PRIOR completed "
                "season; tier is a band on that. A minutes/points proxy for role, "
                "not a fitted impact coefficient. None = no prior season (rookie)."
            ),
        }
        end = injury_data_through()
        if end and as_of > end:
            payload["warnings"] = [
                f"as_of_date is past the end of the injury log ({end}). Injuries for "
                "this date are UNKNOWN, not zero. Report that; do not report an "
                "empty list as though nobody is hurt."
            ]
        return payload

    def player_splits(self, player_name: str, back_to_back: bool = False) -> dict:
        # Without game logs we cannot compute a true back-to-back split, so we
        # say so rather than inventing one.
        today_season = season_end_year(date.today())
        for season in (today_season - 1, today_season - 2):
            avg = player_season_averages(player_name, season)
            if avg:
                avg["source"] = self.name
                if back_to_back:
                    avg["b2b_pts_avg"] = None
                    avg["b2b_unavailable"] = (
                        "Back-to-back splits need per-game logs, which are not on "
                        "main yet. Season averages only."
                    )
                return avg
        return {
            "source": self.name,
            "error": f"player not found in nba_stats: {player_name}",
        }

    def schedule(self, as_of_date: str, days_ahead: int = 1) -> dict:
        """Fixtures tipping off after as_of. See slate_as_of for why that is safe."""
        return slate_as_of(parse_date(as_of_date), days_ahead)

    def player_features(
        self, player_name: str, matchup_id: str, as_of_date: str
    ) -> dict:
        """Trailing form for one player going into one game. See player_features_as_of.

        Note this is per-GAME, unlike `player_splits`, which serves prior-season
        averages. The two answer different questions and read different files.
        """
        away, home, game_date = parse_matchup_id(matchup_id)
        payload = player_features_as_of(
            player_name,
            away,
            home,
            game_date,
            parse_date(as_of_date),
        )
        payload["source"] = self.name
        return payload

    def betting_line(self, matchup_id: str, as_of_date: str) -> dict:
        # The week-5 branch had a second betting_line with no as-of gate, reading
        # a separate 2025-26 odds extract. Both are retired: one gated accessor,
        # one odds file, used by the agent and the eval harness alike.
        away, home, game_date = parse_matchup_id(matchup_id)
        return closing_line(away, home, game_date, parse_date(as_of_date))

    def team_form(self, team_abbr: str, as_of_date: str, last_n: int = 10) -> dict:
        """Current rolling strength from game logs, or a reason it is unavailable."""
        form = team_form_as_of(team_abbr, parse_date(as_of_date), last_n)
        if form is None:
            return {
                "source": self.name,
                "team": normalize_abbr(team_abbr),
                "as_of": as_of_date,
                "unavailable": (
                    "No games played this season before as_of (opening week), or no "
                    f"game logs for season {season_end_year(parse_date(as_of_date))}. "
                    "Fall back to prior-season ratings; do not guess."
                ),
            }
        form["source"] = self.name
        return form


def get_source(kind: str):
    if kind == "mock":
        return MockSource()
    if kind == "real":
        if not INJURY_CSV.exists():
            raise SystemExit(
                "Real data not found. Run `git pull` to get data/raw from main, "
                "or use --source mock."
            )
        return CsvSource()
    raise ValueError(f"unknown source: {kind!r} (expected 'mock' or 'real')")
