import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained LSTM model and scaler
SCALER_PATH = "main/scaler.pkl"
LSTM_MODEL_PATH = "main/lstm_traffic_predictor_tf.keras"

scaler = joblib.load(SCALER_PATH)
model = load_model(LSTM_MODEL_PATH)

def predict_future_conditions(latency, bandwidth, utilization):
    """Predicts future network parameter changes using LSTM."""
    
    # Prepare input in the required shape
    recent_data = np.array([[latency, bandwidth, utilization]])

    # Normalize input for LSTM
    recent_data_scaled = scaler.transform(recent_data)
    recent_data_scaled = np.expand_dims(recent_data_scaled, axis=0)  # Reshape for LSTM model

    # Predict changes
    predicted_changes = model.predict(recent_data_scaled)

    # Convert back to original scale
    predicted_changes = scaler.inverse_transform(predicted_changes)[0]

    return predicted_changes  # Returns [predicted_latency_change, predicted_bandwidth_change, predicted_utilization_change]
