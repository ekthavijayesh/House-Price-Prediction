
from pathlib import Path
import pandas as pd

# Search the project folder and its subfolders for CSV files
project_folder = Path.cwd()

csv_files = list(project_folder.rglob("bengaluru_house_prices.csv"))

print("Current project folder:", project_folder)
print("\nCSV files found:")

for file in csv_files:
    print(file)

if csv_files:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
else:
    print("\nCSV not found inside the project folder.")
    print("Please check the file's location in VS Code Explorer.")
