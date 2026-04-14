import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import sys

# Add project root to path so we can import src modules if run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_aligner import align_features, TARGET_COL

# Setup robust paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'models')
data_path = os.path.join(BASE_DIR, 'data', 'spg.csv')

# create models directory if not there
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# load dataset
df = pd.read_csv(data_path)

df = df.drop_duplicates()

# Use the feature aligner to ensure consistent feature order and schema
print("Aligning features...")
X = align_features(df)
y = df[TARGET_COL]

# split data and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# save model and scaler
joblib.dump(model, os.path.join(models_dir, 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

print(f"Model and scaler saved successfully in '{models_dir}' directory.")
