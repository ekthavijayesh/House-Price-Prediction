
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned data
df = pd.read_csv("data/cleaned_house_prices.csv")

print("Original cleaned dataset:", df.shape)

# 2. Remove clearly suspicious records for this first model
# These are initial screening rules, not universal property rules.
df = df[
    (df["total_sqft"] / df["bhk"] >= 300)
    & (df["total_sqft"] / df["bhk"] <= 3000)
    & (df["bhk"] <= 10)
    & (df["bath"] <= 10)
].copy()

print("After basic outlier screening:", df.shape)

# 3. Define input features and target
features = ["location", "total_sqft", "bhk", "bath"]
X = df[features]
y = df["price"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Convert location text into numerical features
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

# 6. Create the Linear Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 7. Train the model
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# 8. Make predictions on unseen test data
predictions = model.predict(X_test)

# 9. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print(f"MAE:  {mae:.2f} lakh")
print(f"RMSE: {rmse:.2f} lakh")
print(f"R² score: {r2:.3f}")

# 10. Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/linear_regression_model.joblib"
)

print("\nModel saved successfully!")
print("File: models/linear_regression_model.joblib")

# 11. Display a few example predictions
results = pd.DataFrame({
    "Actual price (lakh)": y_test.iloc[:10].values,
    "Predicted price (lakh)": predictions[:10]
})

print("\nSample predictions:")
print(results.round(2).to_string(index=False))
