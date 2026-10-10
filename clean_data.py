
from pathlib import Path
import pandas as pd

# 1. Find the CSV file automatically
csv_files = list(Path.cwd().rglob("bengaluru_house_prices.csv"))

if not csv_files:
    raise FileNotFoundError("Could not find the Bengaluru CSV file.")

df = pd.read_csv(csv_files[0])

print("Original dataset shape:", df.shape)

# 2. Drop columns not required for our first model
df = df.drop(
    columns=["area_type", "availability", "society", "balcony"],
    errors="ignore"
)

# 3. Remove rows with missing essential values
df = df.dropna(subset=["location", "size", "bath", "price"]).copy()

# 4. Convert BHK details into numbers
# Example: "3 BHK" becomes 3; "4 Bedroom" becomes 4
df["bhk"] = pd.to_numeric(
    df["size"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

# 5. Convert square footage into numeric values
def convert_sqft(value):
    text = str(value).strip()

    # Handle ranges such as "1200-1500"
    if "-" in text:
        parts = text.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return None

    # Handle regular values such as "1200" or "1000.5"
    try:
        return float(text)
    except ValueError:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# 6. Convert bathrooms and price to numeric values
df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# 7. Remove rows where essential numeric values are invalid
df = df.dropna(
    subset=["location", "bhk", "total_sqft", "bath", "price"]
).copy()

# 8. Remove impossible or invalid values
df = df[
    (df["bhk"] > 0)
    & (df["total_sqft"] > 0)
    & (df["bath"] > 0)
    & (df["price"] > 0)
].copy()

# 9. Remove duplicate records
df = df.drop_duplicates().copy()

# 10. Save the cleaned dataset
output_file = Path("data") / "cleaned_house_prices.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("Remaining missing values:")
print(df.isnull().sum())
print("\nSaved cleaned data to:", output_file)
print("\nFirst 5 cleaned rows:")
print(df.head())
