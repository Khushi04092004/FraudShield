import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create a simple dummy model
X = np.random.rand(1000, 5)  # 5 features: amount, v1, v2, v3, v4
y = np.random.randint(0, 2, 1000)  # Binary classification: 0=legitimate, 1=fraudulent

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'fraud_detection_model.pkl')
print("Dummy model created and saved as 'fraud_detection_model.pkl'")