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
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from agent.teams import abbr_from_nickname, full_name, normalize_abbr

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = REPO_ROOT / "data" / "mock"
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLE_DIR = REPO_ROOT / "data" / "samples"

INJURY_CSV = RAW_DIR / "injury_data_2016_2025" / "injury_data.csv"
TEAM_SUMMARY_CSV = RAW_DIR / "nba_stats_1947_present" / "Team Summaries.csv"
PLAYER_PER_GAME_CSV = RAW_DIR / "nba_stats_1947_present" / "Player Per Game.csv"


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
    if not INJURY_CSV.exists():
        return ()
    rows: list[tuple[date, str, str, str, str]] = []
    with INJURY_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            abbr = abbr_from_nickname(r["Team"])
            if not abbr:
                continue
            note = r["Notes"].strip()
            acquired = r["Acquired"].strip()
            relinquished = r["Relinquished"].strip()
            if relinquished:
                rows.append((parse_date(r["Date"]), abbr, relinquished, "out", note))
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
        active.append(inj)

    return sorted(active, key=lambda i: i["published"], reverse=True)


@lru_cache(maxsize=1)
def _team_summaries() -> dict[tuple[int, str], dict]:
    if not TEAM_SUMMARY_CSV.exists():
        return {}
    table: dict[tuple[int, str], dict] = {}
    with TEAM_SUMMARY_CSV.open(newline="") as f:
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
    with PLAYER_PER_GAME_CSV.open(newline="") as f:
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
    with path.open(newline="") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def _all_game_logs() -> tuple[dict, ...]:
    """Every season we have on disk. Rest is a within-season question, but
    head-to-head history reaches back across seasons."""
    games: list[dict] = []
    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.glob("game_logs_*.csv")):
            with path.open(newline="") as f:
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
        with path.open() as f:
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
