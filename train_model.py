import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('f1_data.csv')
df = df.dropna(subset=['GridPosition', 'Position'])
df['Won'] = (df['Position'] == 1).astype(int)

# Much stronger weight on recent seasons
# 2024 data counts 8x more than 2021
df['Weight'] = df['Year'].map({
    2021: 1,
    2022: 2,
    2023: 4,
    2024: 8
})

# Encode teams, drivers, circuits
df['TeamEncoded']    = pd.factorize(df['TeamName'])[0]
df['DriverEncoded']  = pd.factorize(df['FullName'])[0]
df['CircuitEncoded'] = pd.factorize(df['Race'])[0]

# Save mappings
team_mapping    = dict(enumerate(pd.factorize(df['TeamName'])[1]))
driver_mapping  = dict(enumerate(pd.factorize(df['FullName'])[1]))
circuit_mapping = dict(enumerate(pd.factorize(df['Race'])[1]))

team_name_to_enc    = {v: k for k, v in team_mapping.items()}
driver_name_to_enc  = {v: k for k, v in driver_mapping.items()}
circuit_name_to_enc = {v: k for k, v in circuit_mapping.items()}

# Add a new feature — how dominant was this team in the most recent season
team_2024_wins = df[df['Year'] == 2024].groupby('TeamName')['Won'].sum()
df['TeamRecentForm'] = df['TeamName'].map(team_2024_wins).fillna(0)

# Add driver 2024 wins
driver_2024_wins = df[df['Year'] == 2024].groupby('FullName')['Won'].sum()
df['DriverRecentForm'] = df['FullName'].map(driver_2024_wins).fillna(0)

features = ['GridPosition', 'TeamEncoded', 'DriverEncoded', 
            'CircuitEncoded', 'TeamRecentForm', 'DriverRecentForm']

X = df[features]
y = df['Won']
w = df['Weight']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=w_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model trained!")
print(f"📊 Accuracy: {accuracy * 100:.1f}%")

pickle.dump(model, open('f1_model.pkl', 'wb'))
pickle.dump(team_name_to_enc, open('teams.pkl', 'wb'))
pickle.dump(driver_name_to_enc, open('drivers.pkl', 'wb'))
pickle.dump(circuit_name_to_enc, open('circuits.pkl', 'wb'))
pickle.dump(team_2024_wins.to_dict(), open('team_form.pkl', 'wb'))
pickle.dump(driver_2024_wins.to_dict(), open('driver_form.pkl', 'wb'))
print("💾 Model saved!")