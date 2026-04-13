import fastf1
fastf1.Cache.enable_cache('cache')
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load(telemetry=False, weather=True, messages=False)
laps = session.laps
pit_in_laps = laps[~laps['PitInTime'].isna()]
pit_out_laps = laps[~laps['PitOutTime'].isna()]
print("Pit in count:", len(pit_in_laps))
print("Pit out count:", len(pit_out_laps))

# check if we have speed trap data without telemetry
print("SpeedST nulls:", laps['SpeedST'].isna().sum(), "out of", len(laps))
