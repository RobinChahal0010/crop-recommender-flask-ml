# train_crop_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1️⃣ Load Dataset
data_path = os.path.join("data", "soil_data.csv")  # Ensure soil_data.csv is here
df = pd.read_csv(data_path)

# 2️⃣ Features & Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']  # Crop names

# 3️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Predictions & Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6️⃣ Save Model
model_dir = os.path.join("models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "crop_model.pkl"))

print("Crop recommendation model trained and saved successfully!")
