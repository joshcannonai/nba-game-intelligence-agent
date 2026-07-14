import requests
import pandas as pd

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
params = {'team': ['ATL', 'NYK']}
r = requests.get(url, params=params, timeout=10)
data = r.json()

records = []
for entry in data['injuries']:
    athlete = entry['athlete']
    records.append({
        'player': athlete['displayName'],
        'team': athlete['team']['abbreviation'],
        'status': entry['status'],
        'injury_type': entry.get('details', {}).get('type', ''),
        'return_date': entry.get('details', {}).get('returnDate', ''),
        'date_reported': entry['date'],
        'comment': entry['shortComment']
    })

df = pd.DataFrame(records)
df.to_csv('data/raw/injuries_test.csv', index=False)
print(df)
