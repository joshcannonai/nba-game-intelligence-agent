import pandas as pd
import os
 
start_year = int(input("Enter start year (example: 2010): "))
end_year = int(input("Enter end year (example: 2020): "))

df = pd.read_csv("data/raw/nba_stats_1947_present/Player Totals.csv")
 
#Filtering data
filtered = df[(df["season"] >= start_year) & (df["season"] <= end_year)]
 
#Save the filtered data into its own separate folder
output_folder = "data/filtered_stats"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/filtered_{start_year}_{end_year}.csv"
filtered.to_csv(output_file, index=False)

print(f"Found {len(filtered)} rows between {start_year} and {end_year}")
print(f"Saved to: {output_file}")
