import fastf1
import os

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()

results = session.results[['DriverNumber', 'FullName', 'TeamName', 'GridPosition', 'Position', 'Points']]
print("\n--- 2024 Bahrain GP Results ---")
print(results.to_string())