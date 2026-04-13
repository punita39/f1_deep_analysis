import fastf1
import os

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Load Bahrain 2024 to see data structure
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load(telemetry=True, weather=True)

# Laps data
laps = session.laps
print(f"Laps columns: {laps.columns.tolist()}")

# Weather data
weather = session.weather_data
print(f"Weather columns: {weather.columns.tolist()}")
print(f"Weather info head: \n{weather.head(3)}")

# Pitstops - lap times where PitInTime and PitOutTime are not null
pitstops = laps.dropna(subset=['PitInTime', 'PitOutTime'])
print(f"Number of pitstops: {len(pitstops)}")
if len(pitstops) > 0:
    pitstops['PitDuration'] = pitstops['PitOutTime'] - pitstops['PitInTime']
    print(f"Sample pit durations: \n{pitstops[['DriverNumber', 'PitDuration']].head(5)}")

