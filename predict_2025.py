import pickle
import numpy as np
import pandas as pd
import xgboost
import warnings

# Suppress sklearn feature name warnings that scramble terminal output
warnings.filterwarnings("ignore", category=UserWarning)

model       = pickle.load(open('24_f1_model.pkl', 'rb'))
teams       = pickle.load(open('24_teams.pkl', 'rb'))
drivers     = pickle.load(open('24_drivers.pkl', 'rb'))
circuits    = pickle.load(open('24_circuits.pkl', 'rb'))

team_static = pickle.load(open('24_team_static.pkl', 'rb'))
driver_form = pickle.load(open('24_driver_form.pkl', 'rb'))
track_hist  = pickle.load(open('24_track_history.pkl', 'rb'))
avg_temps   = pickle.load(open('24_avg_track_temp.pkl', 'rb'))

driver_gap       = pickle.load(open('24_driver_quali_gap.pkl', 'rb'))
latest_champ_pos = pickle.load(open('24_latest_champ_pos.pkl', 'rb'))

RACE = "Abu Dhabi Grand Prix"

# (INTERNAL) The script automatically calculates and stores the temperature in this required variable
TARGET_TRACK_TEMP = avg_temps.get(RACE, 30.0)

import fastf1
import logging

fastf1.Cache.enable_cache('scratch')
logging.getLogger("fastf1").setLevel(logging.ERROR)

print(f"\n[API] Fetching Live Qualifying results for {RACE} via fastf1...")
try:
    # Attempt to load the true 2025 qualifying data
    session = fastf1.get_session(2025, RACE, 'Q')
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results.head(10)
    
    race_entry = []
    for i, row in results.iterrows():
        # Get the driver's fastest Q session time
        q_time_td = row['Q3']
        if pd.isnull(q_time_td): q_time_td = row['Q2']
        if pd.isnull(q_time_td): q_time_td = row['Q1']
        
        q_time_sec = q_time_td.total_seconds() if pd.notnull(q_time_td) else 90.0
        
        race_entry.append({
            "driver": row['FullName'],
            "team": row['TeamName'],
            "grid": int(row['Position']),
            "q_time": round(q_time_sec, 3)
        })
    print(f"[OK] Successfully loaded Pole Sitter: {race_entry[0]['driver']} ({race_entry[0]['q_time']}s)\n")
except Exception as e:
    print(f"[FAIL] Failed to fetch {RACE} live via fastf1: {e}")
    print("Please ensure your RACE string matches the official F1 Calendar name, or check your internet connection!")
    exit()

# ============================================================
# 🏆 UPDATE CHAMPIONSHIP STANDINGS BEFORE EACH RACE
# (positions after the previous race)
# ============================================================
champ_2025_overrides = {
    "Lando Norris": 1.0,
    "Max Verstappen": 2.0,
    "Oscar Piastri": 3.0,
    "George Russell": 4.0,
    "Charles Leclerc": 5.0,
    "Lewis Hamilton": 6.0,
    "Kimi Antonelli": 7.0,
    "Alexander Albon": 8.0,
    "Carlos Sainz": 9.0,
    "Fernando Alonso": 10.0
}
# ============================================================

# Auto calculate QualiGapToTeammate from actual q_times
team_times = {}
for e in race_entry:
    team_times.setdefault(e['team'], []).append(e['q_time'])

for e in race_entry:
    times = team_times[e['team']]
    if len(times) == 2:
        teammate_time = [t for t in times if t != e['q_time']]
        e['q_gap'] = round(e['q_time'] - teammate_time[0], 3) if teammate_time else 0.0
    else:
        e['q_gap'] = 0.0

# ============================================================

circuit_enc = circuits.get(RACE, -1)
if circuit_enc == -1:
    print(f"Warning: Circuit '{RACE}' not found in training data!")
    exit()

print(f"\n{RACE} - Advanced Win Prediction")
print(f"Simulated Track Temperature: {TARGET_TRACK_TEMP:.1f}°C\n")

print(f"{'Driver':<25} {'Team':<20} {'Grid':>4}   {'Win Chance':>10}")
print("-" * 65)

# Calculate global medians for fallbacks in case of new drivers/teams
fallback_pit_time = np.median([v['TeamPitTime'] for v in team_static.values()])
fallback_top_speed = np.median([v['TeamTopSpeed'] for v in team_static.values()])

results = []
for entry in race_entry:
    driver = entry['driver']
    team = entry['team']
    grid = entry['grid']
    
    # Base encodings
    driver_enc = drivers.get(driver, 0)
    team_enc = teams.get(team, 0)
    
    # Form and Static parameters (Team and Driver)
    t_stats = team_static.get(team, {'TeamRecentWins': 0.0, 'TeamTopSpeed': fallback_top_speed, 'TeamPitTime': fallback_pit_time})
    team_rec_form = t_stats.get('TeamRecentWins', 0.0)
    team_top_speed = t_stats.get('TeamTopSpeed', fallback_top_speed)
    team_pit_time = t_stats.get('TeamPitTime', fallback_pit_time)
    
    driver_rec_form = driver_form.get(driver, 0.0)
    
    # Track historical performance (Raw Time)
    trk_stats = track_hist.get((driver, RACE), {'AvgTrackPosition': 10.0, 'AvgQualiLapTime': 85.0})
    if isinstance(trk_stats, dict):
        avg_track_pos = trk_stats.get('AvgTrackPosition', 10.0)
        avg_quali_time = trk_stats.get('AvgQualiLapTime', 85.0)
    else:
        # Fallback
        avg_track_pos = 10.0
        avg_quali_time = 85.0

    # Use REAL 2025 values (overrides historical averages)
    champ_pos       = champ_2025_overrides.get(driver, latest_champ_pos.get(driver, 10.0))
    gap_to_teammate = entry['q_gap']         # Auto calculated above ✅
    avg_quali_time  = entry['q_time']        # Real 2025 quali time ✅
    
    # Create DataFrame with proper feature names to prevent Scikit-learn warnings!
    import pandas as pd
    col_names = ['GridPosition', 'TeamEncoded', 'DriverEncoded', 'CircuitEncoded', 
                 'TeamRecentForm', 'DriverRecentForm', 'TeamTopSpeed', 'TeamPitTime', 
                 'AvgTrackPosition', 'TrackTemp', 'QualiLapTime', 'QualiGapToTeammate', 'ChampionshipPosition']
                 
    features_df = pd.DataFrame([[
        grid, team_enc, driver_enc, circuit_enc,
        team_rec_form, driver_rec_form, team_top_speed, team_pit_time,
        avg_track_pos, TARGET_TRACK_TEMP,
        avg_quali_time, gap_to_teammate, champ_pos
    ]], columns=col_names)

    proba = model.predict_proba(features_df)
    prob  = proba[0][1] * 100 if proba.shape[1] == 2 else 0.0
    results.append((driver, team, grid, prob))

# Normalize output percentages
total   = sum(r[3] for r in results) or 1
results = [(r[0], r[1], r[2], (r[3]/total)*100) for r in results]
results.sort(key=lambda x: x[3], reverse=True)

for driver, team, grid, prob in results:
    bar = "#" * int(prob / 2)
    print(f"{driver:<25} {team:<20} {grid:>4}   {prob:>6.1f}%  {bar}")

print(f"\nPredicted Winner: {results[0][0]}")