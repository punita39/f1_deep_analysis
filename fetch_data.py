import fastf1
import pandas as pd
import os
from fastf1.ergast import Ergast

# Diagnostic Tool for F1 Predictor v2
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')
ergast = Ergast()

def diagnostic_check():
    print("--- F1 Predictor v2: Diagnostic Check ---\n")
    
    # 1. Check Ergast Standings
    print("1. Checking Ergast Standings API...")
    try:
        standings = ergast.get_driver_standings(season=2024, round=1)
        print(f"   Success: Found {len(standings.content[0])} drivers in standings.\n")
    except Exception as e:
        print(f"   Error: Standings API failed: {e}\n")

    # 2. Check Quali Sector Data
    print("2. Checking Qualifying Sector Data (Bahrain 2024)...")
    try:
        session = fastf1.get_session(2024, 'Bahrain', 'Q')
        session.load(telemetry=False, weather=False)
        best_lap = session.laps.pick_fastest()
        print(f"   Success: Fastest Quali Lap Sector Times:")
        print(f"   S1: {best_lap['Sector1Time']}")
        print(f"   S2: {best_lap['Sector2Time']}")
        print(f"   S3: {best_lap['Sector3Time']}\n")
    except Exception as e:
        print(f"   Error: Quali Data load failed: {e}\n")

    # 3. Check Race DNF Status
    print("3. Checking Race Status Codes (Australia 2024 - Max DNF)...")
    try:
        session = fastf1.get_session(2024, 'Australia', 'R')
        session.load(telemetry=False, weather=False)
        results = session.results[session.results['Status'] != 'Finished']
        print(f"   Success: First few DNF Statuses:")
        print(results[['FullName', 'Status']].head(3).to_string(index=False))
        print("")
    except Exception as e:
        print(f"   Error: Race Data load failed: {e}\n")

    print("Diagnostic Complete.")

if __name__ == "__main__":
    diagnostic_check()