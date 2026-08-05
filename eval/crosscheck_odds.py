"""Checks whether our odds file has closing lines or opening lines.

Compares 10 games from our file against an independent source that clearly
labels "Open_Line" vs individual sportsbook lines (which are close to closing).
Whichever one our number is closer to tells us which line type we have.
"""
import pandas as pd

PRIMARY_FILE = "data/raw/odds/primary/nba_2008-2026.csv"
CROSSCHECK_FILE = "data/raw/odds/crosscheck/2014-15/vegas.txt"
SEASON = 2015
N_GAMES = 10

TEAM_ABBR = {
    "orlando": "orl", "new orleans": "no", "dallas": "dal", "san antonio": "sa",
    "houston": "hou", "l.a. lakers": "lal", "milwaukee": "mil", "charlotte": "cha",
    "philadelphia": "phi", "indiana": "ind", "brooklyn": "bkn", "boston": "bos",
    "washington": "wsh", "miami": "mia", "atlanta": "atl", "toronto": "tor",
    "minnesota": "min", "memphis": "mem", "chicago": "chi", "new york": "ny",
}

primary = pd.read_csv(PRIMARY_FILE)
primary = primary[primary["season"] == SEASON]

cross = pd.read_csv(CROSSCHECK_FILE)
cross["team"] = cross["Team"].str.lower().map(TEAM_ABBR)
cross["opp"] = cross["OppTeam"].str.lower().map(TEAM_ABBR)
book_cols = ["Pinnacle_ML", "5dimes_ML", "Heritage_ML", "Bovada_ML", "Betonline_ML"]
cross["closing_ml"] = cross[book_cols].mean(axis=1)

rows = []
for _, g in primary.sort_values(["date", "away", "home"]).iterrows():
    if len(rows) >= N_GAMES:
        break
    match = cross[(cross["Date"] == g["date"]) & (cross["team"] == g["away"]) & (cross["opp"] == g["home"])]
    if match.empty:
        continue

    ours = g["moneyline_away"]
    opening = match.iloc[0]["Open_Line_ML"]
    closing = match.iloc[0]["closing_ml"]

    rows.append({
        "date": g["date"], "away": g["away"], "home": g["home"],
        "our_line": ours, "opening_line": opening, "closing_line": round(closing, 1),
        "closer_to": "OPENING" if abs(ours - opening) < abs(ours - closing) else "CLOSING",
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
print(f"\n{(table['closer_to'] == 'CLOSING').sum()} of {len(table)} games are closer to CLOSING.")