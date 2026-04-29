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
            results['RoundNumber'] = event['RoundNumber']
            
            # Get Quali data
            try:
                q_session = fastf1.get_session(year, event['EventName'], 'Q')
                q_session.load(telemetry=False, weather=False, messages=False)
                q_cols = [c for c in ['Q1', 'Q2', 'Q3'] if c in q_session.results.columns]
                q_results = q_session.results[['DriverNumber'] + q_cols].copy()
                
                def get_quali_time(row):
                    for q in reversed(q_cols):
                        if pd.notna(row[q]):
                            val = row[q]
                            return val.total_seconds() if hasattr(val, 'total_seconds') else np.nan
                    return np.nan
                
                q_results['QualiLapTime'] = q_results.apply(get_quali_time, axis=1)
                results = results.merge(q_results[['DriverNumber', 'QualiLapTime']], on='DriverNumber', how='left')
                
                # Gap calculation
                results['TeamFastestQuali'] = results.groupby('TeamName')['QualiLapTime'].transform('min')
                results['QualiGapToTeammate'] = results['QualiLapTime'] - results['TeamFastestQuali']
                results.drop(columns=['TeamFastestQuali'], inplace=True)
            except Exception as qe:
                results['QualiLapTime'] = np.nan
                results['QualiGapToTeammate'] = np.nan

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

# Calculate ChampionshipPositions
df = df.sort_values(by=['Year', 'RoundNumber'])
df['CumPoints'] = df.groupby(['Year', 'FullName'])['Points'].cumsum()
# PreRacePoints drops their current race points so it measures their standing BEFORE the race
df['PreRacePoints'] = df.groupby(['Year', 'FullName'])['CumPoints'].shift(1).fillna(0)
df['ChampionshipPosition'] = df.groupby(['Year', 'RoundNumber'])['PreRacePoints'].rank(ascending=False, method='min')

# Fill unrecorded data with the medians so we don't lose rows
df['TopSpeed'] = df['TopSpeed'].fillna(df['TopSpeed'].median())
df['MedianPitTime'] = df['MedianPitTime'].fillna(df['MedianPitTime'].median())
df['TrackTemp'] = df['TrackTemp'].fillna(df['TrackTemp'].median())

# Fix: Drivers who crash in Quali shouldn't get the global median (which makes them look like track record holders)
# They should get the slowest time of that session.
df['QualiLapTime'] = df['QualiLapTime'].fillna(df.groupby(['Year', 'Race'])['QualiLapTime'].transform('max'))
df['QualiGapToTeammate'] = df['QualiGapToTeammate'].fillna(df.groupby(['Year', 'Race'])['QualiGapToTeammate'].transform('max'))


df.to_csv('f1_data.csv', index=False)
print("\nAll data saved to f1_data.csv!")
print(f"Total rows: {len(df)}")