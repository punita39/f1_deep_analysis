import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('f1_data.csv')
df = df.dropna(subset=['GridPosition', 'Position'])

# 3. Track History (Calculated BEFORE filtering 2024 to prevent leakage!)
driver_track_history = df[df['Year'] <= 2023].groupby(['FullName', 'Race']).agg(
    AvgTrackPosition=('Position', 'mean'),
    AvgQualiLapTime=('QualiLapTime', 'mean')
).reset_index()

# Filter purely for 2024 data for training and recent form
df = df[df['Year'] == 2024].copy()

df['Won'] = (df['Position'] == 1).astype(int)

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

# 1. Team 2024 spec properties
team_2024 = df.groupby('TeamName').agg(
    TeamTopSpeed=('TopSpeed', 'median'),
    TeamPitTime=('MedianPitTime', 'median'),
    TeamRecentWins=('Won', 'sum')
).reset_index()

df['TeamTopSpeed']   = df['TeamName'].map(team_2024.set_index('TeamName')['TeamTopSpeed']).fillna(df['TopSpeed'].median())
df['TeamPitTime']    = df['TeamName'].map(team_2024.set_index('TeamName')['TeamPitTime']).fillna(df['MedianPitTime'].median())
df['TeamRecentForm'] = df['TeamName'].map(team_2024.set_index('TeamName')['TeamRecentWins']).fillna(0)

# 2. Driver 2024 recent form
driver_2024_wins = df.groupby('FullName')['Won'].sum()
df['DriverRecentForm'] = df['FullName'].map(driver_2024_wins).fillna(0)

# 2.1 Driver 2024 average quali gap to teammate
driver_2024_gap = df.groupby('FullName')['QualiGapToTeammate'].mean()

# 2.2 Latest Championship Position (no duplicate)
latest_round = df['RoundNumber'].max()
latest_champ_pos = df[df['RoundNumber'] == latest_round].set_index('FullName')['ChampionshipPosition']

# 3. Track History
# Merging the historical (<=2023) track data computed prior to the 2024 filter


df = df.merge(driver_track_history, on=['FullName', 'Race'], how='left')

df['AvgTrackPosition'] = df['AvgTrackPosition'].fillna(10.0).clip(1, 15)
df['AvgQualiLapTime']  = df['AvgQualiLapTime'].fillna(df['QualiLapTime'].median())

# Feature Selection
features = [
    'GridPosition',
    'TeamEncoded', 'DriverEncoded', 'CircuitEncoded',
    'TeamRecentForm',
    'DriverRecentForm',
    'TeamTopSpeed',
    'TeamPitTime',
    'AvgTrackPosition',
    'TrackTemp',
    'QualiLapTime',
    'QualiGapToTeammate',
    'ChampionshipPosition'
]

# Ensure no NaNs in features
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df['Won']

# Split train/test randomly since we only have 1 year of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with scale_pos_weight to handle the 1 winner vs 19 losers imbalance
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=19,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)
print(f"Model trained strictly on 2024 parameters!")
print(f"Test Accuracy: {accuracy * 100:.1f}%")

# Save model and encodings uniquely for the 2025 predictor
pickle.dump(model,               open('24_f1_model.pkl', 'wb'))
pickle.dump(team_name_to_enc,    open('24_teams.pkl', 'wb'))
pickle.dump(driver_name_to_enc,  open('24_drivers.pkl', 'wb'))
pickle.dump(circuit_name_to_enc, open('24_circuits.pkl', 'wb'))

# Save feature dictionaries
team_static = team_2024.set_index('TeamName').to_dict('index')
pickle.dump(team_static,               open('24_team_static.pkl', 'wb'))
pickle.dump(driver_2024_wins.to_dict(), open('24_driver_form.pkl', 'wb'))

# Save average historical track temperature
avg_track_temp = df.groupby('Race')['TrackTemp'].mean().to_dict()
pickle.dump(avg_track_temp, open('24_avg_track_temp.pkl', 'wb'))

# Save track history
track_history_dict = driver_track_history.set_index(
    ['FullName', 'Race']
)[['AvgTrackPosition', 'AvgQualiLapTime']].to_dict('index')
pickle.dump(track_history_dict, open('24_track_history.pkl', 'wb'))

# Save new feature proxies
pickle.dump(driver_2024_gap.to_dict(),  open('24_driver_quali_gap.pkl', 'wb'))
pickle.dump(latest_champ_pos.to_dict(), open('24_latest_champ_pos.pkl', 'wb'))

# Print feature importances
importances = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
})
importances.sort_values(by='Importance', ascending=False, inplace=True)
print("\nFeature Importances:")
print(importances.to_string(index=False))

print("\nModel trained exclusively on 2024 data and saved with '24_' prefix!")
