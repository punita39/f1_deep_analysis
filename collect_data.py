import fastf1
import pandas as pd
import os
import numpy as np
from fastf1.ergast import Ergast
from tqdm import tqdm
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')
ergast = Ergast()

def categorize_status(status):
    status = str(status).lower()
    if any(word in status for word in ['accident', 'collision', 'spun', 'crash', 'damage']):
        return 'Collision'
    elif any(word in status for word in ['engine', 'gearbox', 'hydraulics', 'electrical', 'brakes', 'mechanical', 'suspension', 'power unit', 'cooling', 'overheating', 'fuel']):
        return 'Mechanical'
    return 'Finished/Other'

def collect_f1_data(years=[2021, 2022, 2023, 2024]):
    all_rows = []
    
    for year in years:
        print(f"\n--- Processing Season: {year} ---")
        schedule = fastf1.get_event_schedule(year)
        # Filter out preseason testing
        schedule = schedule[schedule['EventFormat'] != 'testing']
        
        for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"Season {year}"):
            race_name = event['EventName']
            round_num = event['RoundNumber']
            
            try:
                # 1. Load Race & Quali Sessions
                race_session = fastf1.get_session(year, race_name, 'R')
                race_session.load(telemetry=False, weather=True)
                
                # Try to load Quali - sometimes it fails for sprint weekends if format changed
                try:
                    quali_session = fastf1.get_session(year, race_name, 'Q')
                    quali_session.load(telemetry=False, weather=False)
                except:
                    # Fallback for sprint formats where Quali might be 'Qualifying' or 'S'
                    quali_session = None

                # 2. Get Championship Standings (Pre-Race)
                # ergast round is 1-indexed. We want standings AFTER prev round.
                try:
                    standings = ergast.get_driver_standings(season=year, round=max(1, round_num-1))
                    standings_df = standings.content[0]
                    # Map to driver name or number
                    points_map = standings_df.set_index('familyName')['points'].to_dict()
                    max_points = standings_df['points'].max() if not standings_df.empty else 0
                except:
                    points_map = {}
                    max_points = 0

                # 3. Process Race Results
                results = race_session.results.copy()
                track_temp = race_session.weather_data['TrackTemp'].mean() if not race_session.weather_data.empty else 30.0
                
                # DNF Mapping
                results['DNF_Type'] = results['Status'].apply(categorize_status)
                results['IsMechanicalDNF'] = (results['DNF_Type'] == 'Mechanical').astype(int)
                
                # 4. Process Laps for Speed and Tires
                laps = race_session.laps
                if laps is not None and not laps.empty:
                    # Top Speed (90th percentile)
                    speeds = laps.groupby('DriverNumber')['SpeedST'].quantile(0.9).to_dict()
                    
                    # Tire Strategy
                    # Dominant compound and avg stint length
                    def get_tire_stats(group):
                        compound = group['Compound'].mode().iloc[0] if not group['Compound'].mode().empty else 'UNKNOWN'
                        stints = group.groupby('Stint').size()
                        avg_stint = stints.mean()
                        return pd.Series({'DominantCompound': compound, 'AvgStintLength': avg_stint})
                    
                    tire_stats = laps.groupby('DriverNumber').apply(get_tire_stats).reset_index()
                else:
                    speeds = {}
                    tire_stats = pd.DataFrame(columns=['DriverNumber', 'DominantCompound', 'AvgStintLength'])

                # 5. Process Qualifying for Deltas and Sectors
                quali_data = {}
                if quali_session is not None and not quali_session.laps.empty:
                    q_laps = quali_session.laps
                    
                    # Gap to Pole
                    fastest_lap_time = q_laps['LapTime'].min()
                    
                    # Process per driver
                    for drv in q_laps['DriverNumber'].unique():
                        drv_laps = q_laps.pick_driver(drv)
                        best_lap = drv_laps.pick_fastest()
                        
                        if pd.notnull(best_lap['LapTime']):
                            q_delta = (best_lap['LapTime'] - fastest_lap_time).total_seconds()
                            s1 = best_lap['Sector1Time'].total_seconds() if pd.notnull(best_lap['Sector1Time']) else np.nan
                            s2 = best_lap['Sector2Time'].total_seconds() if pd.notnull(best_lap['Sector2Time']) else np.nan
                            s3 = best_lap['Sector3Time'].total_seconds() if pd.notnull(best_lap['Sector3Time']) else np.nan
                            
                            quali_data[drv] = {
                                'QualiDelta': q_delta,
                                'S1Time': s1,
                                'S2Time': s2,
                                'S3Time': s3
                            }
                
                # 6. Build Row Data
                for _, row in results.iterrows():
                    drv_num = row['DriverNumber']
                    drv_name = row['LastName'] # Used for standings map
                    
                    # Championship Pressure
                    pts = float(points_map.get(drv_name, 0))
                    pts_gap = max_points - pts
                    
                    # Build record
                    record = {
                        'Year': year,
                        'RoundNumber': round_num,
                        'Race': race_name,
                        'FullName': row['FullName'],
                        'TeamName': row['TeamName'],
                        'GridPosition': row['GridPosition'],
                        'Position': row['Position'],
                        'Status': row['Status'],
                        'Points': row['Points'],
                        'Won': 1 if row['Position'] == 1 else 0,
                        'TrackTemp': track_temp,
                        'IsWet': 1 if (race_session.laps['IsAccurate'].sum() > 0 and 'Wet' in race_session.laps['Compound'].unique()) else 0,
                        'PointsGapToLeader': pts_gap,
                        'TopSpeed': speeds.get(drv_num, np.nan),
                        'IsMechanicalDNF': row['IsMechanicalDNF']
                    }
                    
                    # Add Tire details
                    t_row = tire_stats[tire_stats['DriverNumber'] == drv_num]
                    if not t_row.empty:
                        record['DominantCompound'] = t_row['DominantCompound'].values[0]
                        record['AvgStintLength'] = t_row['AvgStintLength'].values[0]
                    else:
                        record['DominantCompound'] = 'UNKNOWN'
                        record['AvgStintLength'] = np.nan
                        
                    # Add Quali details
                    q_vals = quali_data.get(drv_num, {})
                    record['QualiDelta'] = q_vals.get('QualiDelta', np.nan)
                    record['S1Time'] = q_vals.get('S1Time', np.nan)
                    record['S2Time'] = q_vals.get('S2Time', np.nan)
                    record['S3Time'] = q_vals.get('S3Time', np.nan)
                    
                    all_rows.append(record)
                    
                print(f" Loaded: {year} {race_name}")
                
            except Exception as e:
                print(f" Failed: {year} {race_name} - {str(e)}")
                continue
                
    df = pd.DataFrame(all_rows)
    
    # Final cleanup & imputations (within race groups)
    # Fill missing QualiDelta with a penalty based on GridPosition
    df['QualiDelta'] = df.groupby(['Year', 'RoundNumber'])['QualiDelta'].transform(lambda x: x.fillna(x.mean() + 2.0))
    
    # Save
    df.to_csv('f1_advanced_data.csv', index=False)
    print(f"\nSaved {len(df)} rows to f1_advanced_data.csv!")

if __name__ == "__main__":
    collect_f1_data()