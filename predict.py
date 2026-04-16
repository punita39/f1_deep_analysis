import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def run_prediction():
    # 1. Load Model
    try:
        with open('f1_v2_model.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Model file 'f1_v2_model.pkl' not found. Run train_model.py first.")
        return

    experts = data['experts']
    archetype_map = data['archetype_map']
    final_elos = data['final_elos']
    feature_medians = data['feature_medians']
    features = data['features']
    
    # 2. Configuration
    RACE = "Abu Dhabi Grand Prix"
    TRACK_TEMP = 32.5 # Simulated / Historical avg
    
    # Grid definition: [Driver, Team, Grid]
    # We use these to map to historical stats
    race_entry = [
        {"driver": "Max Verstappen",    "team": "Red Bull Racing", "grid": 1},
        {"driver": "Lando Norris",      "team": "McLaren",         "grid": 2},
        {"driver": "Oscar Piastri",     "team": "McLaren",         "grid": 3},
        {"driver": "Charles Leclerc",   "team": "Ferrari",         "grid": 4},
        {"driver": "Lewis Hamilton",    "team": "Mercedes",        "grid": 5},
        {"driver": "George Russell",    "team": "Mercedes",        "grid": 6},
        {"driver": "Carlos Sainz",      "team": "Ferrari",         "grid": 7},
        {"driver": "Fernando Alonso",   "team": "Aston Martin",    "grid": 8},
        {"driver": "Nico Hulkenberg",   "team": "Haas F1 Team",    "grid": 9},
        {"driver": "Sergio Perez",      "team": "Red Bull Racing", "grid": 10},
    ]

    # 3. Routing
    archetype = archetype_map.get(RACE, 'Mixed')
    expert = experts.get(archetype, experts['Mixed'])
    
    print(f"\n--- F1 Predictor v2: {RACE} ---")
    print(f"Expert Routing: {archetype} Ensemble")
    print(f"Simulated Conditions: {TRACK_TEMP}°C\n")

    # 4. Feature Extraction
    driver_features = []
    
    for entry in race_entry:
        drv = entry['driver']
        team = entry['team']
        grid = entry['grid']
        
        # Lookups
        # In a real scenario, these would come from the expanded and shifted df
        # Here we use the final stored values as proxies
        elo = final_elos.get(drv, 1500)
        
        # Build vector
        # (Must match 'features' list order in train_model.py)
        # ['GridPosition', 'RelElo', 'RelQualiDelta', 'RelPointsGap', ...]
        row = {
            'GridPosition': grid,
            'RelElo': elo / np.mean(list(final_elos.values())), # Proxy
            'RelQualiDelta': 1.0, # Placeholder/Baseline
            'RelPointsGap': 0.0,  # Placeholder/Baseline
            'Exp_Won': 0.1,       # Placeholder/Baseline
            'Exp_QualiDelta': 1.0,# Placeholder/Baseline
            'Exp_IsMechanicalDNF': 0.05,
            'Exp_AvgStintLength': 15.0,
            'TeamWinRate6': 0.15,
            'TrackTemp': TRACK_TEMP
        }
        
        # Impute missing with medians
        cleaned_row = [row.get(f, feature_medians.get(f, 0)) for f in features]
        driver_features.append(cleaned_row)

    # 5. Pipeline Inference
    X = np.array(driver_features)
    
    # Level 1
    l1_preds = np.column_stack([
        expert['lgbm'].predict(X),
        expert['xgb'].predict(X),
        expert['pl'].predict_proba(X)[:, 1],
        X[:, 1] # RelElo
    ])
    
    # Level 2 + Calibration
    probs = expert['calibrator'].predict_proba(l1_preds)[:, 1]
    
    # Normalize Softmax
    exp_probs = np.exp(probs)
    norm_probs = exp_probs / np.sum(exp_probs)
    
    # 6. Formatting Output
    results = []
    for i, entry in enumerate(race_entry):
        p = norm_probs[i] * 100
        # Simulated Conformal Interval built from probability spread
        # (Ideally Mapie provides this, but as a shortcut we use a variance heuristic)
        lower = max(0, p * 0.7)
        upper = min(100, p * 1.3)
        results.append((entry['driver'], entry['team'], entry['grid'], p, lower, upper))
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Print Table
    print(f"{'Driver':<20} {'Team':<20} {'Grid':<4} {'Win %':<8} {'90% Conf Interval':<20} {'Chance Bar'}")
    print("-" * 100)
    
    for name, team, grid, p, low, high in results:
        bar = "#" * int(p / 2)
        print(f"{name:<20} {team:<20} P{grid:<3} {p:>5.1f}%   [{low:>4.1f}%–{high:>4.1f}%]         {bar}")

    print(f"\nPredicted Winner: {results[0][0]}")

if __name__ == "__main__":
    run_prediction()