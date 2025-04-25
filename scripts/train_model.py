import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# Load dataset
df = pd.read_csv("data/german_credit_data.csv")

# Preprocessing
df.dropna(inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Target encoding
target_col = "Risk"
df_encoded[target_col] = df[target_col].map({"good": 1, "bad": 0})

# Features and target
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Save feature columns
feature_cols = X.columns.tolist()
os.makedirs("models", exist_ok=True)
pd.Series(feature_cols).to_csv("models/feature_columns.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardization (for LIME compatibility)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "models/credit_risk_model.pkl")

# Evaluation
y_pred = model.predict(X_test_scaled)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
prec = round(precision_score(y_test, y_pred) * 100, 2)
rec = round(recall_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred) * 100, 2)
roc = round(roc_auc_score(y_test, y_pred) * 100, 2)
cm = confusion_matrix(y_test, y_pred).tolist()

# Save metrics
metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "roc_auc": roc,
    "confusion_matrix": cm
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Model training complete.")