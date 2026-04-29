import pickle
import numpy as np
import warnings

# Suppress sklearn feature name warnings that scramble terminal output
warnings.filterwarnings("ignore", category=UserWarning)

model       = pickle.load(open('f1_model.pkl', 'rb'))
teams       = pickle.load(open('teams.pkl', 'rb'))
drivers     = pickle.load(open('drivers.pkl', 'rb'))
circuits    = pickle.load(open('circuits.pkl', 'rb'))

team_static = pickle.load(open('team_static.pkl', 'rb'))
driver_form = pickle.load(open('driver_form.pkl', 'rb'))
track_hist  = pickle.load(open('track_history.pkl', 'rb'))
avg_temps   = pickle.load(open('avg_track_temp.pkl', 'rb'))

driver_gap       = pickle.load(open('driver_quali_gap.pkl', 'rb'))
latest_champ_pos = pickle.load(open('latest_champ_pos.pkl', 'rb'))

RACE = "Saudi Arabian Grand Prix"

# (INTERNAL) The script automatically calculates and stores the temperature in this required variable
TARGET_TRACK_TEMP = avg_temps.get(RACE, 30.0)

# A generic historical race grid for 2024 analysis
race_entry = [
    {"driver": "Max Verstappen",    "team": "Red Bull Racing", "grid": 1},
    {"driver": "Sergio Perez",      "team": "Red Bull Racing", "grid": 2},
    {"driver": "Charles Leclerc",   "team": "Ferrari",         "grid": 3},
    {"driver": "Carlos Sainz",      "team": "Ferrari",         "grid": 4},
    {"driver": "Lando Norris",      "team": "McLaren",         "grid": 5},
    {"driver": "Oscar Piastri",     "team": "McLaren",         "grid": 6},
    {"driver": "George Russell",    "team": "Mercedes",        "grid": 7},
    {"driver": "Lewis Hamilton",    "team": "Mercedes",        "grid": 8},
    {"driver": "Fernando Alonso",   "team": "Aston Martin",    "grid": 9},
    {"driver": "Lance Stroll",      "team": "Aston Martin",    "grid": 10},
]

circuit_enc = circuits.get(RACE, -1)
if circuit_enc == -1:
    print(f"Warning: Circuit '{RACE}' not found in training data!")
    exit()

print(f"\n{RACE} - 2024 Baseline Win Prediction")
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

    # For 2024, use the calculated statistics directly from the pickle files
    champ_pos       = latest_champ_pos.get(driver, 10.0)
    gap_to_teammate = driver_gap.get(driver, 0.0)
    
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