import fastf1
import pandas as pd
import os

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

all_races = []

# 2021 to 2025 — all completed seasons
years = [2021, 2022, 2023, 2024]

for year in years:
    schedule = fastf1.get_event_schedule(year)
    for _, event in schedule.iterrows():
        try:
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load(telemetry=False, weather=False, messages=False)
            results = session.results[['FullName', 'TeamName', 'GridPosition', 'Position', 'Points']]
            results['Year'] = year
            results['Race'] = event['EventName']
            all_races.append(results)
            print(f"✅ Loaded: {year} {event['EventName']}")
        except Exception as e:
            print(f"⏭️ Skipped: {year} {event['EventName']} — {e}")

df = pd.concat(all_races)
df.to_csv('f1_data.csv', index=False)
print("\n✅ All data saved to f1_data.csv!")
print(f"Total rows: {len(df)}")