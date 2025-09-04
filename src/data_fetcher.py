import requests
import pandas as pd
import os

# Your credentials and the API endpoint
API_KEY = 'ca5f5661459c42c1b8d69e2bd266f51d'

# --- THIS IS THE UPDATED LINE ---
# We've added '?season=2025' to get all matches for the current season.
URL = 'https://api.football-data.org/v4/competitions/PL/matches?season=2025'

# Set up the headers
headers = {'X-Auth-Token': API_KEY}

response = requests.get(URL, headers=headers)

if response.status_code == 200:
    print("Success! Data fetched for the entire season.")
    data = response.json()
    matches = data['matches']

    processed_matches = []
    for match in matches:
        if match['status'] == 'FINISHED':
            processed_matches.append({
                'date': match['utcDate'],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_goals': match['score']['fullTime']['home'],
                'away_goals': match['score']['fullTime']['away'],
                'result': match['score']['winner'],
            })

    df = pd.DataFrame(processed_matches)
    save_path = 'data/raw/premier_league_matches.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Data processed and saved to '{save_path}'")

    # Let's check the LAST 5 rows to see the most recent matches!
    print("Most recent matches:")
    print(df.tail())

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print(f"Response: {response.text}")