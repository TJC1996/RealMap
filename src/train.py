import os
import pandas as pd
import numpy as np
import joblib
import argparse
import boto3
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------
# 1. Argument Parsing (SageMaker)
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, default="data/commericalnj.csv")
parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
args = parser.parse_args()

# ------------------------
# 2. Load Data
# ------------------------
df = pd.read_csv(args.train_data, low_memory=False)

# Preprocessing
df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")
df["Sale Year"] = df["Sale Date"].dt.year
df["Sale Month"] = df["Sale Date"].dt.month

# Define Features & Target
features = ["Sq. Ft.", "Acreage", "Sale Year", "Latitude", "Longitude"]
target = "Sale Price"

df = df.dropna(subset=features + [target])

X = df[features]
y = np.log1p(df[target])  # Log transform for better modeling

# ------------------------
# 3. Train-Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# 4. Train Model
# ------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")

# ------------------------
# 5. Save Model (for SageMaker Deployment)
# ------------------------
joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
print("Model saved successfully!")
