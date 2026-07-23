import os
import glob
import shutil
from datetime import datetime
import pandas as pd
 
cutoff_date = input("Enter cutoff date (YYYY-MM-DD): ")
cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
 
input_folder = "data/raw"
snapshot_path = f"data/snapshots/{cutoff_date}"
os.makedirs(snapshot_path, exist_ok=True)
 
# 1. Player box scores
box_score_files = glob.glob(f"{input_folder}/player_box_scores_*.csv")
kept = 0
for file_path in box_score_files:
    filename = os.path.basename(file_path)
    date_text = filename.replace("player_box_scores_", "").replace(".csv", "")
    file_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    if file_date <= cutoff:
        shutil.copy(file_path, f"{snapshot_path}/{filename}")
        kept += 1
 
print(f"Player box scores: copied {kept} of {len(box_score_files)} daily files")
 
# 2. Injuries - filter rows by date_reported
injuries_file = f"{input_folder}/injuries.csv"
if os.path.exists(injuries_file):
    df = pd.read_csv(injuries_file)
    df["date_reported"] = pd.to_datetime(df["date_reported"])
    filtered = df[df["date_reported"].dt.date <= cutoff]
    filtered.to_csv(f"{snapshot_path}/injuries.csv", index=False)
    print(f"Injuries: kept {len(filtered)} of {len(df)} rows")
 
# 3. Season schedule
schedule_files = glob.glob(f"{input_folder}/season_schedule_*.csv")
for file_path in schedule_files:
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize(None)
    filtered = df[df["start_time"].dt.date <= cutoff]
    filtered.to_csv(f"{snapshot_path}/{filename}", index=False)
    print(f"Season schedule: kept {len(filtered)} of {len(df)} games")
 
print(f"Done! Snapshot saved at: {snapshot_path}")
