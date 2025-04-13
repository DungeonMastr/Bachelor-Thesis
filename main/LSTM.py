import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import joblib

print("Stop 1")
# Enable GPU acceleration if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Stop 2")
# Load traffic data from CSV
df = pd.read_csv("LSTM_traffic.csv")
print("Stop 3")
# Select relevant features
df = df[['timestamp', 'source_node', 'destination_node', 'latency_ms', 'bandwidth_mbps', 'utilization_percent']]
print("Stop 4")
# Compute **parameter changes** instead of absolute values
df['latency_change'] = df['latency_ms'].diff().fillna(0)
df['bandwidth_change'] = df['bandwidth_mbps'].diff().fillna(0)
df['utilization_change'] = df['utilization_percent'].diff().fillna(0)
print("Stop 5")
# Normalize data
scaler = MinMaxScaler()
df[['latency_change', 'bandwidth_change', 'utilization_change']] = scaler.fit_transform(
    df[['latency_change', 'bandwidth_change', 'utilization_change']]
)
print("Stop 6")
# Save scaler for later use in ACO
joblib.dump(scaler, "main/scaler.pkl")
print("Stop 7")
# Define time-series window size
SEQ_LENGTH = 10
print("Stop 8")
# Create sequences for LSTM training
X, y = [], []
for i in range(len(df) - SEQ_LENGTH):
    X.append(df[['latency_change', 'bandwidth_change', 'utilization_change']].iloc[i:i+SEQ_LENGTH].values.astype(np.float32))
    y.append(df[['latency_change', 'bandwidth_change', 'utilization_change']].iloc[i+SEQ_LENGTH].values.astype(np.float32))

X, y = np.array(X), np.array(y, dtype=np.float32)
print("Stop 9")
# Split into training and validation sets
split_idx = int(0.85 * len(X))
X_train, X_val, y_train, y_val = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
print("Stop 10")
# Define the LSTM Model using Keras with optimized architecture
model = Sequential([
    LSTM(units=48, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2]), name="LSTM_Layer1"),
    BatchNormalization(),
    Dropout(0.1),
    LSTM(units=24, return_sequences=False, name="LSTM_Layer2"),
    Dropout(0.1),
    Dense(3, activation='tanh')  # Output: Predicted changes in latency, bandwidth, and utilization
])
print("Stop 11")
# Compile model with improved optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
print("Stop 12")
# Train the model using early stopping for best performance
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])
print("Stop 13")
# Save the trained model
model.save("main/lstm_traffic_predictor_tf.keras")

print("LSTM Model Trained and Saved with Predicted Parameter Adjustments.")




