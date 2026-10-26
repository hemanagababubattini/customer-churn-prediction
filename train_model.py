
import pandas as pd

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove customerID because it is an identifier
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.value_counts())

# Convert categorical features into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Convert target values into numbers
y = y.map({"No": 0, "Yes": 1})

print("\nEncoded features shape:", X.shape)
print("\nFirst 5 encoded rows:")
print(X.head())

print("\nTarget values after encoding:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train-Test Split Results ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts())

print("\nTesting churn distribution:")
print(y_test.value_counts())

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train-Test Split Results ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts())

print("\nTesting churn distribution:")
print(y_test.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\n--- Model Training Completed ---")

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", round(accuracy * 100, 2), "%")

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Confusion Matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
print("\n--- Random Forest Results ---")
print("Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")

print(classification_report(
    y_test,
    rf_pred,
    target_names=["No Churn", "Churn"]
))

import os
import joblib

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save the trained Random Forest model
joblib.dump(rf_model, "models/churn_model.pkl")

print("\nRandom Forest model saved successfully!")
