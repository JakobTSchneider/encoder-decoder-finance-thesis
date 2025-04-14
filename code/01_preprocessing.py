# Block 1: Imports, Seeds und konfigurierbare Parameter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as dt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Activation, Dot, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
import os
import random
import time

# Reproduzierbarkeit einstellen
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

if hasattr(tf, 'config'):
    tf.config.experimental.enable_op_determinism()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Zeitmessung Callback
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_times = []
        self.training_start = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_start_time)
        self.times.append(time.time())

    def on_train_end(self, logs={}):
        self.total_training_time = time.time() - self.training_start

# Hilfsfunktionen
def directional_accuracy(y_true, y_pred):
    """Berechnet die direktionale Genauigkeit (Trendrichtung)."""
    direction_true = np.diff(y_true.flatten())
    direction_pred = np.diff(y_pred.flatten())
    correct_direction = (np.sign(direction_true) == np.sign(direction_pred))
    return np.mean(correct_direction) * 100

def analyze_by_volatility(actual, predicted, volatility_threshold=0.01):
    """Segmentiert Daten nach Volatilität und berechnet Metriken."""
    # Volatilität berechnen
    daily_returns = np.diff(actual.flatten()) / actual.flatten()[:-1]
    daily_volatility = np.abs(daily_returns)

    # Segmentieren
    high_vol_indices = daily_volatility >= volatility_threshold
    low_vol_indices = daily_volatility < volatility_threshold

    # Array-Länge anpassen
    high_vol_indices = np.append(high_vol_indices, False)
    low_vol_indices = np.append(low_vol_indices, False)

    # Daten filtern
    high_vol_actual = actual[high_vol_indices]
    high_vol_pred = predicted[high_vol_indices]
    low_vol_actual = actual[low_vol_indices]
    low_vol_pred = predicted[low_vol_indices]

    # Metriken berechnen
    high_vol_metrics = {}
    low_vol_metrics = {}

    if len(high_vol_actual) > 0:
        high_vol_metrics['mae'] = mean_absolute_error(high_vol_actual, high_vol_pred)
        high_vol_metrics['rmse'] = np.sqrt(mean_squared_error(high_vol_actual, high_vol_pred))
        if len(high_vol_actual) > 1:
            high_vol_metrics['da'] = directional_accuracy(high_vol_actual, high_vol_pred)

    if len(low_vol_actual) > 0:
        low_vol_metrics['mae'] = mean_absolute_error(low_vol_actual, low_vol_pred)
        low_vol_metrics['rmse'] = np.sqrt(mean_squared_error(low_vol_actual, low_vol_pred))
        if len(low_vol_actual) > 1:
            low_vol_metrics['da'] = directional_accuracy(low_vol_actual, low_vol_pred)

    return high_vol_metrics, low_vol_metrics

def create_sequences(data, input_len, output_len):
    """Erstellt Input- und Output-Sequenzen für Zeitreihenmodellierung."""
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

# Konfigurierbare Parameter
input_seq_length = 60  # Hier ändern für verschiedene Input-Längen (z.B. 20, 60, 90, 120)
output_seq_length = 20  # Hier ändern für verschiedene Output-Längen (z.B. 5, 10, 20)

# Daten laden
print("Lade S&P 500 Daten...")
end_date = dt.datetime(2024, 4, 1)
start_date = dt.datetime(2018, 4, 1)
data = yf.download("^GSPC", start=start_date, end=end_date)
print(f"Daten geladen. Form: {data.shape}")

# Datenvorverarbeitung
df = data[['Close']].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Sequenzen erstellen
X, y = create_sequences(scaled_data, input_seq_length, output_seq_length)
print(f"Sequenzdaten erstellt. X Form: {X.shape}, y Form: {y.shape}")

# Daten aufteilen
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Daten für das Modell aufbereiten
X_train = X_train.reshape(-1, input_seq_length, 1)
X_val = X_val.reshape(-1, input_seq_length, 1)
X_test = X_test.reshape(-1, input_seq_length, 1)

# Decoder-Inputs für Teacher Forcing
decoder_input_train = np.zeros((len(X_train), output_seq_length, 1))
decoder_input_val = np.zeros((len(X_val), output_seq_length, 1))
decoder_input_test = np.zeros((len(X_test), output_seq_length, 1))

# Ersten Zeitschritt mit dem letzten bekannten Wert initialisieren
decoder_input_train[:, 0, 0] = X_train[:, -1, 0]
decoder_input_val[:, 0, 0] = X_val[:, -1, 0]
decoder_input_test[:, 0, 0] = X_test[:, -1, 0]

# Output-Shape anpassen
y_train = y_train.reshape(-1, output_seq_length, 1)
y_val = y_val.reshape(-1, output_seq_length, 1)
y_test = y_test.reshape(-1, output_seq_length, 1)
