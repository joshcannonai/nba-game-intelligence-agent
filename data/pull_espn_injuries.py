import requests
import pandas as pd

teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET',
         'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
         'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS',
         'TOR', 'UTA', 'WAS']

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
all_records = []

for team in teams:
    r = requests.get(url, params={'team': team}, timeout=10)
    data = r.json()
    for entry in data.get('injuries', []):
        athlete = entry['athlete']
        all_records.append({
            'player': athlete['displayName'],
            'team': athlete['team']['abbreviation'],
            'status': entry['status'],
            'injury_type': entry.get('details', {}).get('type', ''),
            'return_date': entry.get('details', {}).get('returnDate', ''),
            'date_reported': entry['date'],
            'comment': entry['shortComment']
        })

df = pd.DataFrame(all_records)
df.to_csv('data/raw/injuries.csv', index=False)
print(f"Pulled {len(df)} injury records across {df['team'].nunique()} teams")
