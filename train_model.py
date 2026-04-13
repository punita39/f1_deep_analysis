import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('f1_data.csv')
df = df.dropna(subset=['GridPosition', 'Position'])
df['Won'] = (df['Position'] == 1).astype(int)

# Much stronger weight on recent seasons
df['Weight'] = df['Year'].map({
    2021: 1,
    2022: 2,
    2023: 4,
    2024: 8
}).fillna(1)

# Encode teams, drivers, circuits
df['TeamEncoded']    = pd.factorize(df['TeamName'])[0]
df['DriverEncoded']  = pd.factorize(df['FullName'])[0]
df['CircuitEncoded'] = pd.factorize(df['Race'])[0]

team_mapping    = dict(enumerate(pd.factorize(df['TeamName'])[1]))
driver_mapping  = dict(enumerate(pd.factorize(df['FullName'])[1]))
circuit_mapping = dict(enumerate(pd.factorize(df['Race'])[1]))

team_name_to_enc    = {v: k for k, v in team_mapping.items()}
driver_name_to_enc  = {v: k for k, v in driver_mapping.items()}
circuit_name_to_enc = {v: k for k, v in circuit_mapping.items()}

# 1. Team's 2024 spec properties (Speed trap proxies Car Specs, PitTime proxies Support Team/Maintenance)
# Also total Support Team Performance represented by TeamRecentWins!
team_2024 = df[df['Year'] == 2024].groupby('TeamName').agg(
    TeamTopSpeed=('TopSpeed', 'median'),
    TeamPitTime=('MedianPitTime', 'median'),
    TeamRecentWins=('Won', 'sum')
).reset_index()

df['TeamTopSpeed'] = df['TeamName'].map(team_2024.set_index('TeamName')['TeamTopSpeed']).fillna(df['TopSpeed'].median())
df['TeamPitTime'] = df['TeamName'].map(team_2024.set_index('TeamName')['TeamPitTime']).fillna(df['MedianPitTime'].median())
df['TeamRecentForm'] = df['TeamName'].map(team_2024.set_index('TeamName')['TeamRecentWins']).fillna(0)

# 2. Driver 2024 recent form
driver_2024_wins = df[df['Year'] == 2024].groupby('FullName')['Won'].sum()
df['DriverRecentForm'] = df['FullName'].map(driver_2024_wins).fillna(0)

# 3. Track History for Driver (Driver performance in specific track)
driver_track_history = df.groupby(['FullName', 'Race'])['Position'].mean().reset_index()
driver_track_history.rename(columns={'Position': 'AvgTrackPosition'}, inplace=True)
df = df.merge(driver_track_history, on=['FullName', 'Race'], how='left')
df['AvgTrackPosition'] = df['AvgTrackPosition'].fillna(10.0) # default fallback

# Feature Selection (Proxy Maps for Requested parameters)
features = [
    'GridPosition',       # Qualifying Match
    'TeamEncoded', 'DriverEncoded', 'CircuitEncoded', 
    'TeamRecentForm',     # Support Team proxy
    'DriverRecentForm', 
    'TeamTopSpeed',       # Car Specs proxy
    'TeamPitTime',        # Maintenance / Pitstop Proxy
    'AvgTrackPosition',   # Specific Track Skill Proxy
    'TrackTemp'           # Weather proxy
]

# Ensure no NaNs in features
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df['Won']
w = df['Weight']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=w_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained with ADVANCED parameters!")
print(f"Accuracy: {accuracy * 100:.1f}%")

# Save outputs and dictionaries for predict.py
pickle.dump(model, open('f1_model.pkl', 'wb'))
pickle.dump(team_name_to_enc, open('teams.pkl', 'wb'))
pickle.dump(driver_name_to_enc, open('drivers.pkl', 'wb'))
pickle.dump(circuit_name_to_enc, open('circuits.pkl', 'wb'))

# Save form and static features dictionaries
team_static = team_2024.set_index('TeamName').to_dict('index')
pickle.dump(team_static, open('team_static.pkl', 'wb'))
pickle.dump(driver_2024_wins.to_dict(), open('driver_form.pkl', 'wb'))

# Save average historical track temperature for each circuit
avg_track_temp = df.groupby('Race')['TrackTemp'].mean().to_dict()
pickle.dump(avg_track_temp, open('avg_track_temp.pkl', 'wb'))

track_history_dict = driver_track_history.set_index(['FullName', 'Race'])['AvgTrackPosition'].to_dict()
pickle.dump(track_history_dict, open('track_history.pkl', 'wb'))

# Print feature importances
importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
importances.sort_values(by='Importance', ascending=False, inplace=True)
print("\nFeature Importances:")
print(importances.to_string(index=False))

print("\nAdvanced Model and features saved!")