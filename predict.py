import pickle
import numpy as np

model       = pickle.load(open('f1_model.pkl', 'rb'))
teams       = pickle.load(open('teams.pkl', 'rb'))
drivers     = pickle.load(open('drivers.pkl', 'rb'))
circuits    = pickle.load(open('circuits.pkl', 'rb'))
team_form   = pickle.load(open('team_form.pkl', 'rb'))
driver_form = pickle.load(open('driver_form.pkl', 'rb'))

RACE = "Bahrain Grand Prix"

race_entry = [
    {"driver": "Oscar Piastri",     "team": "McLaren",         "grid": 1},
    {"driver": "Lando Norris",      "team": "McLaren",         "grid": 2},
    {"driver": "Max Verstappen",    "team": "Red Bull Racing", "grid": 3},
    {"driver": "George Russell",    "team": "Mercedes",        "grid": 4},
    {"driver": "Charles Leclerc",   "team": "Ferrari",         "grid": 5},
    {"driver": "Lewis Hamilton",    "team": "Ferrari",         "grid": 6},
    {"driver": "Carlos Sainz",      "team": "Williams",        "grid": 7},
    {"driver": "Fernando Alonso",   "team": "Aston Martin",    "grid": 8},
    {"driver": "Pierre Gasly",      "team": "Alpine",          "grid": 9},
    {"driver": "Lance Stroll",      "team": "Aston Martin",    "grid": 10},
]

circuit_enc = circuits.get(RACE, -1)
if circuit_enc == -1:
    print(f"⚠️ Circuit '{RACE}' not found!")
    exit()

print(f"\n🏎️  {RACE} — Win Probability Prediction\n")
print(f"{'Driver':<25} {'Team':<20} {'Grid':>4}   {'Win Chance':>10}")
print("-" * 65)

results = []
for entry in race_entry:
    driver_enc      = drivers.get(entry['driver'], 0)
    team_enc        = teams.get(entry['team'], 0)
    team_rec_form   = team_form.get(entry['team'], 0)
    driver_rec_form = driver_form.get(entry['driver'], 0)

    features = np.array([[
        entry['grid'],
        team_enc,
        driver_enc,
        circuit_enc,
        team_rec_form,
        driver_rec_form
    ]])

    proba = model.predict_proba(features)
    prob  = proba[0][1] * 100 if proba.shape[1] == 2 else 0.0
    results.append((entry['driver'], entry['team'], entry['grid'], prob))

# Normalize
total   = sum(r[3] for r in results) or 1
results = [(r[0], r[1], r[2], (r[3]/total)*100) for r in results]
results.sort(key=lambda x: x[3], reverse=True)

for driver, team, grid, prob in results:
    bar = "█" * int(prob / 2)
    print(f"{driver:<25} {team:<20} {grid:>4}   {prob:>6.1f}%  {bar}")

print(f"\n🏆 Predicted Winner: {results[0][0]}")