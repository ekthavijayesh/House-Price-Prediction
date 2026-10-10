
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

print("Dataset shape:", df.shape)

# 2. Display basic statistics
print("\nStatistical summary:")
print(df[["total_sqft", "bhk", "bath", "price"]].describe())

# Create a folder to save graphs
Path("plots").mkdir(exist_ok=True)

# 3. House price distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="price", bins=50)
plt.title("Distribution of House Prices")
plt.xlabel("Price (₹ lakh)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("plots/price_distribution.png")
plt.show()
plt.close()

# 4. Area versus price
# Use a sample to make the graph easier to read
sample = df.sample(
    n=min(2000, len(df)),
    random_state=42
)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=sample,
    x="total_sqft",
    y="price",
    alpha=0.5
)
plt.title("House Area vs Price")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (₹ lakh)")
plt.tight_layout()
plt.savefig("plots/area_vs_price.png")
plt.show()
plt.close()

# 5. Bedrooms versus price
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df[df["bhk"] <= 6],
    x="bhk",
    y="price"
)
plt.title("Bedrooms vs House Price")
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Price (₹ lakh)")
plt.tight_layout()
plt.savefig("plots/bedrooms_vs_price.png")
plt.show()
plt.close()

# 6. Bathrooms versus price
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df[df["bath"] <= 6],
    x="bath",
    y="price"
)
plt.title("Bathrooms vs House Price")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price (₹ lakh)")
plt.tight_layout()
plt.savefig("plots/bathrooms_vs_price.png")
plt.show()
plt.close()

print("\nEDA completed successfully!")
print("Graphs saved inside the plots folder.")
