"""Team identity mapping.

The three data sources name teams three different ways:
  injury log  -> nickname only ("Blazers")
  nba_stats   -> abbreviation ("POR") + full name ("Portland Trail Blazers")
  matchup ids -> abbreviation ("POR")

Everything internal keys off the abbreviation.
"""

from __future__ import annotations

# abbr -> (full name, injury-log nickname)
TEAMS: dict[str, tuple[str, str]] = {
    "ATL": ("Atlanta Hawks", "Hawks"),
    "BOS": ("Boston Celtics", "Celtics"),
    "BRK": ("Brooklyn Nets", "Nets"),
    "CHI": ("Chicago Bulls", "Bulls"),
    "CHO": ("Charlotte Hornets", "Hornets"),
    "CLE": ("Cleveland Cavaliers", "Cavaliers"),
    "DAL": ("Dallas Mavericks", "Mavericks"),
    "DEN": ("Denver Nuggets", "Nuggets"),
    "DET": ("Detroit Pistons", "Pistons"),
    "GSW": ("Golden State Warriors", "Warriors"),
    "HOU": ("Houston Rockets", "Rockets"),
    "IND": ("Indiana Pacers", "Pacers"),
    "LAC": ("Los Angeles Clippers", "Clippers"),
    "LAL": ("Los Angeles Lakers", "Lakers"),
    "MEM": ("Memphis Grizzlies", "Grizzlies"),
    "MIA": ("Miami Heat", "Heat"),
    "MIL": ("Milwaukee Bucks", "Bucks"),
    "MIN": ("Minnesota Timberwolves", "Timberwolves"),
    "NOP": ("New Orleans Pelicans", "Pelicans"),
    "NYK": ("New York Knicks", "Knicks"),
    "OKC": ("Oklahoma City Thunder", "Thunder"),
    "ORL": ("Orlando Magic", "Magic"),
    "PHI": ("Philadelphia 76ers", "76ers"),
    "PHO": ("Phoenix Suns", "Suns"),
    "POR": ("Portland Trail Blazers", "Blazers"),
    "SAC": ("Sacramento Kings", "Kings"),
    "SAS": ("San Antonio Spurs", "Spurs"),
    "TOR": ("Toronto Raptors", "Raptors"),
    "UTA": ("Utah Jazz", "Jazz"),
    "WAS": ("Washington Wizards", "Wizards"),
}

# Abbreviations that differ between sources. nba_api and Basketball-Reference
# disagree on three teams; normalize everything to the BRK/CHO/PHO spellings
# used by the nba_stats CSVs.
ABBR_ALIASES = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}

NICKNAME_TO_ABBR = {nick: abbr for abbr, (_, nick) in TEAMS.items()}


def normalize_abbr(abbr: str) -> str:
    a = abbr.strip().upper()
    return ABBR_ALIASES.get(a, a)


def abbr_from_nickname(nickname: str) -> str | None:
    return NICKNAME_TO_ABBR.get(nickname.strip())


def full_name(abbr: str) -> str:
    return TEAMS[normalize_abbr(abbr)][0]
