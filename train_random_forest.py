
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

print("Original cleaned dataset:", df.shape)

# 2. Apply the same outlier filters as Linear Regression
df = df[
    (df["total_sqft"] / df["bhk"] >= 300)
    & (df["total_sqft"] / df["bhk"] <= 3000)
    & (df["bhk"] <= 10)
    & (df["bath"] <= 10)
].copy()

print("After outlier screening:", df.shape)

# 3. Select features and target
features = ["location", "total_sqft", "bhk", "bath"]
X = df[features]
y = df["price"]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Preprocess location and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["location"]
        ),
        (
            "numeric",
            "passthrough",
            ["total_sqft", "bhk", "bath"]
        )
    ]
)

# 6. Create Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )
    )
])

# 7. Train the model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# 8. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRANDOM FOREST EVALUATION")
print(f"MAE:  {mae:.2f} lakh")
print(f"RMSE: {rmse:.2f} lakh")
print(f"R² score: {r2:.3f}")

# 9. Save the model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/random_forest_model.joblib"
)

print("\nRandom Forest model saved successfully!")
print("File: models/random_forest_model.joblib")

# 10. Display sample predictions
results = pd.DataFrame({
    "Actual price (lakh)": y_test.iloc[:10].values,
    "Predicted price (lakh)": predictions[:10]
})

print("\nSample predictions:")
print(results.round(2).to_string(index=False))
