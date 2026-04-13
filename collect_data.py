import fastf1
import pandas as pd
import os
import numpy as np

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

all_races = []
years = [2021, 2022, 2023, 2024]

for year in years:
    schedule = fastf1.get_event_schedule(year)
    for _, event in schedule.iterrows():
        try:
            session = fastf1.get_session(year, event['EventName'], 'R')
            # Load weather but skip heavy telemetry
            session.load(telemetry=False, weather=True, messages=False)
            
            results = session.results[['DriverNumber', 'FullName', 'TeamName', 'GridPosition', 'Position', 'Points']].copy()
            results['Year'] = year
            results['Race'] = event['EventName']
            
            # Weather: Track Temperature
            if not session.weather_data.empty:
                results['TrackTemp'] = session.weather_data['TrackTemp'].mean()
            else:
                results['TrackTemp'] = np.nan
                
            # Pitstops and Car Topspeed
            if session.laps is not None and not session.laps.empty:
                laps = session.laps
                
                # SpeedST (Speed trap logic) - we take the 90th percentile to avoid anomalies
                speeds = laps.groupby('DriverNumber')['SpeedST'].apply(lambda x: x.quantile(0.9)).reset_index()
                speeds = speeds.rename(columns={'SpeedST': 'TopSpeed'})
                
                # Pitstop duration
                pit_laps = laps.dropna(subset=['PitInTime', 'PitOutTime']).copy()
                if not pit_laps.empty:
                    pit_laps['PitDuration'] = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds()
                    pitstops = pit_laps.groupby('DriverNumber')['PitDuration'].median().reset_index()
                    pitstops = pitstops.rename(columns={'PitDuration': 'MedianPitTime'})
                else:
                    pitstops = pd.DataFrame(columns=['DriverNumber', 'MedianPitTime'])
                
                results = results.merge(speeds, on='DriverNumber', how='left')
                results = results.merge(pitstops, on='DriverNumber', how='left')
            else:
                results['TopSpeed'] = np.nan
                results['MedianPitTime'] = np.nan
                
            all_races.append(results)
            print(f"Loaded: {year} {event['EventName']}")
        except Exception as e:
            print(f"Skipped: {year} {event['EventName']} - {e}")

df = pd.concat(all_races)

# Fill unrecorded data with the medians so we don't lose rows
df['TopSpeed'] = df['TopSpeed'].fillna(df['TopSpeed'].median())
df['MedianPitTime'] = df['MedianPitTime'].fillna(df['MedianPitTime'].median())
df['TrackTemp'] = df['TrackTemp'].fillna(df['TrackTemp'].median())

df.to_csv('f1_data.csv', index=False)
print("\nAll data saved to f1_data.csv!")
print(f"Total rows: {len(df)}")