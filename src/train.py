import pandas as pd
import numpy as np
import joblib
import tarfile
import os
import time
import boto3
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ===============================
# 🔐 1. Load Environment Variables
# ===============================
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/trained-model")
PREPROCESSOR_PREFIX = os.getenv("PREPROCESSOR_PREFIX", "models/preprocessor")

# ===============================
# 📥 2. Load & Clean Dataset
# ===============================
file_path = "data/commericalnj.csv"
df = pd.read_csv(file_path, low_memory=False)

# Convert 'Sale Date' to datetime and extract features
df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")
df["Sale Year"] = df["Sale Date"].dt.year
df["Sale Month"] = df["Sale Date"].dt.month
df["Sale Month Sine"] = np.sin(2 * np.pi * df["Sale Month"] / 12)
df["Sale Month Cosine"] = np.cos(2 * np.pi * df["Sale Month"] / 12)

# Convert numeric columns
numeric_cols = ["Sq. Ft.", "Yr. Built", "Acreage", "Latitude", "Longitude", "Sale Price", "Total Assmnt", "Taxes 1"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Filter out unrealistic sales and outliers
df = df[df["Sale Price"] >= 500000]  
df = df[df["Sale Price"] <= df["Sale Price"].quantile(0.95)]  

# Feature engineering
df["Building Age"] = pd.Timestamp.today().year - df["Yr. Built"]
df["Log Sale Price"] = np.log1p(df["Sale Price"])

# ===============================
# 🔧 3. Define Features & Target
# ===============================
numerical_features = ["Sq. Ft.", "Acreage", "Building Age", "Sale Year",
                      "Sale Month Sine", "Sale Month Cosine",
                      "Latitude", "Longitude", "Total Assmnt", "Taxes 1"]
categorical_features = ["Municipality", "Property Class", "Type/Use", "Neigh"]

X = df[numerical_features + categorical_features].copy()
y = df["Log Sale Price"].copy()

# Remove any infinite values and drop rows with NaNs
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 🔄 4. Preprocessing & Model Training
# ===============================
# Create the preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

# Fit the preprocessor on the training data and transform both training and test sets
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train the XGBoost Model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", 
    random_state=42, 
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1
)
xgb_model.fit(X_train_transformed, y_train)

# ===============================
# 🎯 5. Evaluate & Save Model
# ===============================
y_test_pred_log = xgb_model.predict(X_test_transformed)
y_test_pred = np.expm1(y_test_pred_log)
y_test_actual = np.expm1(y_test)

test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
test_r2 = r2_score(y_test_actual, y_test_pred)
print(f"🏡 Test RMSE: {test_rmse:,.2f}")
print(f"📈 Test R²: {test_r2:.2f}")

# Save the preprocessor using joblib
joblib.dump(preprocessor, "preprocessor.joblib")
print("✅ Preprocessor saved to preprocessor.joblib")

# Save the XGBoost model in XGBoost’s native format to a temporary file
model_file = "xgboost-model.deprecated"
xgb_model.save_model(model_file)
print(f"✅ XGBoost model saved to {model_file}")

# Create a SageMaker-compatible tarball for the model
tar_model_path = "model.tar.gz"
with tarfile.open(tar_model_path, "w:gz") as tar:
    # The SageMaker XGBoost container expects a file named exactly "xgboost-model.bst"
    tar.add(model_file, arcname="xgboost-model.bst")
print(f"✅ Model tarball saved to {tar_model_path}")

# ===============================
# 🚀 6. Upload Artifacts to S3
# ===============================
s3 = boto3.client("s3")

# Upload the model tarball
s3.upload_file(tar_model_path, S3_BUCKET, f"{MODEL_PREFIX}/model.tar.gz")
print(f"✅ Model tarball uploaded to s3://{S3_BUCKET}/{MODEL_PREFIX}/model.tar.gz")

# Upload the preprocessor file
s3.upload_file("preprocessor.joblib", S3_BUCKET, f"{PREPROCESSOR_PREFIX}/preprocessor.joblib")
print(f"✅ Preprocessor uploaded to s3://{S3_BUCKET}/{PREPROCESSOR_PREFIX}/preprocessor.joblib")
