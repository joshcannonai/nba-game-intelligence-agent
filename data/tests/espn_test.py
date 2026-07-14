import requests

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
params = {'team': ['ATL', 'NYK']}  # test with a couple teams first
r = requests.get(url, params=params, timeout=10)
print(r.status_code)
print(r.json())
