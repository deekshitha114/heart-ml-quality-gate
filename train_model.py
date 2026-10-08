import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Read dataset
df = pd.read_csv("heart.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save model
with open("heart_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Create metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

# Save metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model trained successfully")
print("Accuracy:", round(accuracy, 4))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
