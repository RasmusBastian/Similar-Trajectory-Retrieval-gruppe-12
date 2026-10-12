from typing import List

import numpy as np
import pandas as pd

# List of edge data files from the 5 days
edge_data_files: List[str] = [
    "./output/edge_data_day3.csv",
    "./output/edge_data_day4.csv",
    "./output/edge_data_day5.csv",
    "./output/edge_data_day6.csv",
    "./output/edge_data_day7.csv",
]

print("\n=== Averaging results across days ===")

matrices: List[pd.DataFrame] = []
for file in edge_data_files:
    df: pd.DataFrame = pd.read_csv(file)
    matrices.append(df)
    print(f"Loaded {file}: shape {df.shape}")

# Start with a copy of the first matrix structure
combined_df: pd.DataFrame = matrices[0].copy()

# Count valid (non -1) entries for each cell across all days
valid_counts: pd.DataFrame = sum(
    [(df.iloc[:, 1:] != -1).astype(int) for df in matrices]
)

# Sum all valid values (replacing -1 with 0 for summation)
total_sum: pd.DataFrame = sum(
    [df.iloc[:, 1:].mask(df.iloc[:, 1:] == -1, 0) for df in matrices]
)

# Calculate averages only for valid entries
averaged_values: pd.DataFrame = total_sum / valid_counts.replace(0, np.nan)

# Create final averaged dataframe
averaged_df: pd.DataFrame = combined_df.copy()
averaged_df.iloc[:, 1:] = averaged_values.fillna(-1)

# Save the averaged results
averaged_df.to_csv("./output/edge_data_averaged.csv", index=False)

print(
    "\n=== Final averaged file saved to ./output/edge_data_averaged.csv ==="
)
print(f"Shape: {averaged_df.shape}")

# Calculate data density (percentage of non-empty cells)
total_cells: int = averaged_df.shape[0] * (averaged_df.shape[1] - 1)
valid_cells: int = (averaged_df.iloc[:, 1:] != -1).sum().sum()
data_density: float = valid_cells / total_cells * 100

print(f"Data density: {data_density:.2f}%")
