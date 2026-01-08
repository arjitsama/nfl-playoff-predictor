import nflreadpy as nfl
import pandas as pd
from pathlib import Path

# Ensure output folder exists
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Load schedules (multi-season so we can train with past results)
sched = nfl.load_schedules(seasons=[2022, 2023, 2024, 2025])

# Load team stats (weekly team stats across seasons)
stats = nfl.load_team_stats(seasons=[2022, 2023, 2024, 2025])

# Convert to pandas if needed (nflreadpy often returns polars)
if not isinstance(sched, pd.DataFrame):
    sched = sched.to_pandas()

if not isinstance(stats, pd.DataFrame):
    stats = stats.to_pandas()

# Preview
print("Schedules preview:")
print(sched.head())
print("\nTeam stats preview:")
print(stats.head())

# Save to CSV
sched.to_csv("data/raw/nfl_schedules_2022_2025.csv", index=False)
stats.to_csv("data/raw/nfl_team_stats_2022_2025.csv", index=False)

print("\nSaved CSVs to data/raw/")
