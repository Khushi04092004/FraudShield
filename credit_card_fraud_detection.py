from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = 'fraud_detection_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print(f"ERROR: Model file not found at {model_path}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('error.html', error="Model not loaded. Please check if the model file exists.")

        # Get form inputs safely
        v1 = float(request.form.get('feature_1', 0))
        v2 = float(request.form.get('feature_2', 0))
        v3 = float(request.form.get('feature_3', 0))
        v4 = float(request.form.get('feature_4', 0))
        amount = float(request.form.get('feature_29', 0))

        # Combine into a NumPy array
        features_array = np.array([[amount, v1, v2, v3, v4]])
        print(f"Received features: {features_array}")

        # Prediction
        prediction_label = model.predict(features_array)[0]

        # Confidence score
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(features_array)[0]) * 100
        else:
            confidence = 0.0

        # Display label
        prediction = "Fraudulent" if prediction_label == 1 else "Legitimate"

        # Prepare feature dictionary for display
        feature_dict = {
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "Amount": amount
        }

        return render_template(
            'result.html',
            prediction=prediction,
            confidence=round(confidence, 2),
            features=feature_dict
        )

    except Exception as e:
        error = f"An error occurred during prediction: {str(e)}"
        print(error)
        return render_template('error.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)

