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

def mean_absolute_percentage_error(y_true, y_pred):
    """Berechnet den Mean Absolute Percentage Error (MAPE)."""
    # Vermeidung von Division durch Null
    mask = y_true.flatten() != 0
    y_true_masked = y_true.flatten()[mask]
    y_pred_masked = y_pred.flatten()[mask]

    # Berechnung des absoluten prozentualen Fehlers
    ape = np.abs((y_true_masked - y_pred_masked) / y_true_masked)

    # Mittelwert der absoluten prozentualen Fehler (in Prozent)
    mape = 100.0 * np.mean(ape)

    return mape

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
        high_vol_metrics['mape'] = mean_absolute_percentage_error(high_vol_actual, high_vol_pred)
        if len(high_vol_actual) > 1:
            high_vol_metrics['da'] = directional_accuracy(high_vol_actual, high_vol_pred)
            high_vol_metrics['r2'] = r2_score(high_vol_actual, high_vol_pred)

    if len(low_vol_actual) > 0:
        low_vol_metrics['mae'] = mean_absolute_error(low_vol_actual, low_vol_pred)
        low_vol_metrics['rmse'] = np.sqrt(mean_squared_error(low_vol_actual, low_vol_pred))
        low_vol_metrics['mape'] = mean_absolute_percentage_error(low_vol_actual, low_vol_pred)
        if len(low_vol_actual) > 1:
            low_vol_metrics['da'] = directional_accuracy(low_vol_actual, low_vol_pred)
            low_vol_metrics['r2'] = r2_score(low_vol_actual, low_vol_pred)

    return high_vol_metrics, low_vol_metrics

def create_sequences(data, input_len, output_len):
    """Erstellt Input- und Output-Sequenzen für Zeitreihenmodellierung."""
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

def build_classic_model(input_seq_length, output_seq_length, latent_dim=180):
    """Erstellt ein klassisches Encoder-Decoder-LSTM-Modell."""
    # Encoder und Decoder Inputs
    encoder_inputs = Input(shape=(input_seq_length, 1))
    decoder_inputs = Input(shape=(output_seq_length, 1))

    # Encoder
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Ausgabeschicht für Multi-Step
    output = TimeDistributed(Dense(1))(decoder_outputs)

    # Modell definieren
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model

def build_hybrid_model(input_seq_length, output_seq_length, latent_dim=100, dropout_rate=0.1):
    """Erstellt ein hybrides Encoder-Decoder-Modell mit Bidirectional LSTM und Attention."""
    # Encoder und Decoder Inputs
    encoder_inputs = Input(shape=(input_seq_length, 1))
    decoder_inputs = Input(shape=(output_seq_length, 1))

    # Encoder mit Bidirectional LSTM
    encoder = Bidirectional(LSTM(latent_dim, dropout=dropout_rate, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    # Zustände für den Decoder
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder = LSTM(latent_dim*2, dropout=dropout_rate, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

    # Attention-Mechanismus
    attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention = Activation('softmax')(attention)
    context = Dot(axes=[2, 1])([attention, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([context, decoder_outputs])

    # Ausgabeschicht
    output = TimeDistributed(Dense(1))(decoder_combined_context)

    # Modell definieren
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model

def evaluate_model(model, X_test, decoder_input_test, y_test, scaler, output_seq_length):
    """Evaluiert das Modell und berechnet verschiedene Metriken."""
    # Inferenzzeit messen
    inference_start_time = time.time()
    predictions = model.predict([X_test, decoder_input_test])
    inference_duration = time.time() - inference_start_time

    # Rücktransformation für Metriken
    predictions_flat = predictions.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_flat)
    y_test_original = scaler.inverse_transform(y_test_flat)

    # Gesamtmetriken
    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mape = mean_absolute_percentage_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    da = directional_accuracy(y_test_original, predictions_original)

    # Metriken nach Prognosehorizont
    predictions_by_step = []
    actual_by_step = []
    horizon_metrics = []

    for step in range(output_seq_length):
        step_preds = predictions[:, step, 0].reshape(-1, 1)
        step_actual = y_test[:, step, 0].reshape(-1, 1)

        # Zurück in die ursprüngliche Skala
        step_preds_original = scaler.inverse_transform(step_preds)
        step_actual_original = scaler.inverse_transform(step_actual)

        predictions_by_step.append(step_preds_original)
        actual_by_step.append(step_actual_original)

        # Metriken für diesen Horizont
        step_mae = mean_absolute_error(step_actual_original, step_preds_original)
        step_rmse = np.sqrt(mean_squared_error(step_actual_original, step_preds_original))
        step_mape = mean_absolute_percentage_error(step_actual_original, step_preds_original)
        step_da = directional_accuracy(step_actual_original, step_preds_original)
        step_r2 = r2_score(step_actual_original, step_preds_original)

        horizon_metrics.append({
            'step': step + 1,
            'mae': step_mae,
            'rmse': step_rmse,
            'mape': step_mape,
            'da': step_da,
            'r2': step_r2
        })

    # Volatilitätsanalyse für ersten Tag
    high_vol_metrics, low_vol_metrics = analyze_by_volatility(
        actual_by_step[0], predictions_by_step[0], 0.01
    )

    # Ergebnisse zusammenfassen
    results = {
        'overall': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'da': da,
            'inference_time': inference_duration,
            'inference_time_per_sample': inference_duration / len(X_test)
        },
        'horizon': horizon_metrics,
        'volatility': {
            'high': high_vol_metrics,
            'normal': low_vol_metrics
        }
    }

    return results, predictions, predictions_by_step, actual_by_step

def train_and_evaluate_for_asset(data, asset_name, asset_volatility, input_len, output_len):
    """Trainiert und evaluiert Modelle für ein bestimmtes Asset und eine bestimmte Konfiguration."""
    print(f"\n===== Test: {asset_name} mit Input={input_len}, Output={output_len} =====")

    # Datenvorverarbeitung
    df = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Sequenzen erstellen
    X, y = create_sequences(scaled_data, input_len, output_len)

    # Datenaufteilung
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Daten aufbereiten
    X_train = X_train.reshape(-1, input_len, 1)
    X_val = X_val.reshape(-1, input_len, 1)
    X_test = X_test.reshape(-1, input_len, 1)

    # Decoder-Inputs
    decoder_input_train = np.zeros((len(X_train), output_len, 1))
    decoder_input_val = np.zeros((len(X_val), output_len, 1))
    decoder_input_test = np.zeros((len(X_test), output_len, 1))

    # Ersten Zeitschritt initialisieren
    decoder_input_train[:, 0, 0] = X_train[:, -1, 0]
    decoder_input_val[:, 0, 0] = X_val[:, -1, 0]
    decoder_input_test[:, 0, 0] = X_test[:, -1, 0]

    # Output-Shape anpassen
    y_train = y_train.reshape(-1, output_len, 1)
    y_val = y_val.reshape(-1, output_len, 1)
    y_test = y_test.reshape(-1, output_len, 1)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # 1. Klassisches Modell trainieren
    print("\nTraining des klassischen Modells...")
    classic_model = build_classic_model(input_len, output_len)
    classic_params = classic_model.count_params()

    time_callback = TimeHistory()
    classic_history = classic_model.fit(
        [X_train, decoder_input_train],
        y_train,
        validation_data=([X_val, decoder_input_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, time_callback],
        verbose=1,
        shuffle=False
    )

    classic_training_time = time_callback.total_training_time

    # Klassisches Modell evaluieren
    print("\nEvaluierung des klassischen Modells...")
    classic_results, classic_predictions, classic_pred_by_step, classic_actual_by_step = evaluate_model(
        classic_model, X_test, decoder_input_test, y_test, scaler, output_len
    )

    # 2. Hybrides Modell trainieren
    print("\nTraining des hybriden Modells...")
    hybrid_model = build_hybrid_model(input_len, output_len)
    hybrid_params = hybrid_model.count_params()

    time_callback = TimeHistory()
    hybrid_history = hybrid_model.fit(
        [X_train, decoder_input_train],
        y_train,
        validation_data=([X_val, decoder_input_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, time_callback],
        verbose=1,
        shuffle=False
    )

    hybrid_training_time = time_callback.total_training_time

    # Hybrides Modell evaluieren
    print("\nEvaluierung des hybriden Modells...")
    hybrid_results, hybrid_predictions, hybrid_pred_by_step, hybrid_actual_by_step = evaluate_model(
        hybrid_model, X_test, decoder_input_test, y_test, scaler, output_len
    )

    # Ergebnisse speichern
    asset_results = {
        'asset_name': asset_name,
        'volatility': asset_volatility,
        'input_length': input_len,
        'output_length': output_len,
        'classic_model': {
            'parameters': classic_params,
            'training_time': classic_training_time,
            'metrics': classic_results
        },
        'hybrid_model': {
            'parameters': hybrid_params,
            'training_time': hybrid_training_time,
            'metrics': hybrid_results
        }
    }

    # Genauigkeitsvergleich für diesen Asset visualisieren
    # R²-Vergleich
    plt.figure(figsize=(20, 10))

    # R² Plot
    plt.subplot(2, 2, 1)
    classic_r2 = [m['r2'] for m in classic_results['horizon']]
    hybrid_r2 = [m['r2'] for m in hybrid_results['horizon']]

    x = range(1, output_len + 1)
    plt.plot(x, classic_r2, 'b-o', label='Klassisches Modell')
    plt.plot(x, hybrid_r2, 'r-o', label='Hybrides Modell')

    plt.title(f'R² Bestimmtheitsmaß: {asset_name} (In={input_len}, Out={output_len})')
    plt.xlabel('Prognosehorizont (Tage)')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(min(classic_r2), min(hybrid_r2))-0.1, 1.0)

    # MAE-Vergleich
    plt.subplot(2, 2, 2)
    classic_mae = [m['mae'] for m in classic_results['horizon']]
    hybrid_mae = [m['mae'] for m in hybrid_results['horizon']]

    plt.plot(x, classic_mae, 'b-o', label='Klassisches Modell')
    plt.plot(x, hybrid_mae, 'r-o', label='Hybrides Modell')

    plt.title(f'MAE: {asset_name} (In={input_len}, Out={output_len})')
    plt.xlabel('Prognosehorizont (Tage)')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # MAPE-Vergleich
    plt.subplot(2, 2, 3)
    classic_mape = [m['mape'] for m in classic_results['horizon']]
    hybrid_mape = [m['mape'] for m in hybrid_results['horizon']]

    plt.plot(x, classic_mape, 'b-o', label='Klassisches Modell')
    plt.plot(x, hybrid_mape, 'r-o', label='Hybrides Modell')

    plt.title(f'MAPE: {asset_name} (In={input_len}, Out={output_len})')
    plt.xlabel('Prognosehorizont (Tage)')
    plt.ylabel('MAPE (%)')
    plt.legend()
    plt.grid(True)

    # Direktionale Genauigkeit
    plt.subplot(2, 2, 4)
    classic_da = [m['da'] for m in classic_results['horizon']]
    hybrid_da = [m['da'] for m in hybrid_results['horizon']]

    plt.plot(x, classic_da, 'b-o', label='Klassisches Modell')
    plt.plot(x, hybrid_da, 'r-o', label='Hybrides Modell')
    plt.axhline(y=50, color='gray', linestyle='--', label='Zufallsniveau (50%)')

    plt.title(f'Direktionale Genauigkeit: {asset_name} (In={input_len}, Out={output_len})')
    plt.xlabel('Prognosehorizont (Tage)')
    plt.ylabel('Direktionale Genauigkeit (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(40, 100)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/asset_comparison_{asset_name.replace(" ", "_").replace("/", "_")}_in{input_len}_out{output_len}.png')
    plt.close()

    # Speicherplatz freigeben
    tf.keras.backend.clear_session()

    return asset_results

def plot_multi_asset_results(df):
    """Erstellt Zusammenfassungsgrafiken für die Asset-Ergebnisse."""
    # Für jede Konfiguration separate Visualisierungen erstellen
    configs = df[['input_length', 'output_length']].drop_duplicates().values.tolist()

    for input_len, output_len in configs:
        config_df = df[(df['input_length'] == input_len) & (df['output_length'] == output_len)]

        # Sortieren nach Volatilität
        config_df = config_df.sort_values(by='volatility')
        assets = config_df['asset'].unique()

        # Hauptmetriken nach Asset und Modelltyp
        fig, axes = plt.subplots(4, 1, figsize=(15, 24))

        # MAE vs. Asset nach Modelltyp
        classic = config_df[config_df['model_type'] == 'Klassisch']
        hybrid = config_df[config_df['model_type'] == 'Hybrid']

        # Breite der Balken für den Barplot
        width = 0.35
        x = np.arange(len(assets))

        # MAE Plot
        axes[0].bar(x - width/2, classic['mae'], width, label='Klassisch', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, hybrid['mae'], width, label='Hybrid', color='red', alpha=0.7)
        axes[0].set_title(f'MAE nach Asset (In={input_len}, Out={output_len})')
        axes[0].set_ylabel('MAE')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(assets, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, axis='y')

        # MAPE Plot
        axes[1].bar(x - width/2, classic['mape'], width, label='Klassisch', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, hybrid['mape'], width, label='Hybrid', color='red', alpha=0.7)
        axes[1].set_title(f'MAPE nach Asset (In={input_len}, Out={output_len})')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(assets, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, axis='y')

        # R² Plot
        axes[2].bar(x - width/2, classic['r2'], width, label='Klassisch', color='blue', alpha=0.7)
        axes[2].bar(x + width/2, hybrid['r2'], width, label='Hybrid', color='red', alpha=0.7)
        axes[2].set_title(f'R² nach Asset (In={input_len}, Out={output_len})')
        axes[2].set_ylabel('R²')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(assets, rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(True, axis='y')

        # Direktionale Genauigkeit Plot
        axes[3].bar(x - width/2, classic['da'], width, label='Klassisch', color='blue', alpha=0.7)
        axes[3].bar(x + width/2, hybrid['da'], width, label='Hybrid', color='red', alpha=0.7)
        axes[3].axhline(y=50, color='gray', linestyle='--', label='Zufallsniveau (50%)')
        axes[3].set_title(f'Direktionale Genauigkeit nach Asset (In={input_len}, Out={output_len})')
        axes[3].set_ylabel('Direktionale Genauigkeit (%)')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(assets, rotation=45, ha='right')
        axes[3].legend()
        axes[3].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/asset_metrics_comparison_in{input_len}_out{output_len}.png')
        plt.close()

        # Differenzen zwischen Modellen berechnen
        diff_data = []
        for asset in assets:
            classic_asset = classic[classic['asset'] == asset].iloc[0]
            hybrid_asset = hybrid[hybrid['asset'] == asset].iloc[0]

            diff_data.append({
                'asset': asset,
                'volatility': classic_asset['volatility'],
                'mae_diff': classic_asset['mae'] - hybrid_asset['mae'],
                'rmse_diff': classic_asset['rmse'] - hybrid_asset['rmse'],
                'mape_diff': classic_asset['mape'] - hybrid_asset['mape'],
                'r2_diff': hybrid_asset['r2'] - classic_asset['r2'],  # Positive Werte bedeuten Hybrid ist besser
                'da_diff': hybrid_asset['da'] - classic_asset['da']   # Positive Werte bedeuten Hybrid ist besser
            })

        diff_df = pd.DataFrame(diff_data)
        diff_df = diff_df.sort_values(by='volatility')

        # Visualisierung der Differenzen
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))

        # MAE Differenz
        axes[0, 0].bar(diff_df['asset'], diff_df['mae_diff'], color=['g' if x > 0 else 'r' for x in diff_df['mae_diff']])
        axes[0, 0].set_title(f'MAE Differenz (Klassisch - Hybrid) (In={input_len}, Out={output_len})')
        axes[0, 0].set_ylabel('MAE Differenz')
        axes[0, 0].axhline(y=0, color='black', linestyle='-')
        axes[0, 0].set_xticklabels(diff_df['asset'], rotation=45, ha='right')
        axes[0, 0].grid(True, axis='y')

        # RMSE Differenz
        axes[0, 1].bar(diff_df['asset'], diff_df['rmse_diff'], color=['g' if x > 0 else 'r' for x in diff_df['rmse_diff']])
        axes[0, 1].set_title(f'RMSE Differenz (Klassisch - Hybrid) (In={input_len}, Out={output_len})')
        axes[0, 1].set_ylabel('RMSE Differenz')
        axes[0, 1].axhline(y=0, color='black', linestyle='-')
        axes[0, 1].set_xticklabels(diff_df['asset'], rotation=45, ha='right')
        axes[0, 1].grid(True, axis='y')

        # MAPE Differenz
        axes[1, 0].bar(diff_df['asset'], diff_df['mape_diff'], color=['g' if x > 0 else 'r' for x in diff_df['mape_diff']])
        axes[1, 0].set_title(f'MAPE Differenz (Klassisch - Hybrid) (In={input_len}, Out={output_len})')
        axes[1, 0].set_ylabel('MAPE Differenz (%)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-')
        axes[1, 0].set_xticklabels(diff_df['asset'], rotation=45, ha='right')
        axes[1, 0].grid(True, axis='y')

        # R² Differenz
        axes[1, 1].bar(diff_df['asset'], diff_df['r2_diff'], color=['g' if x > 0 else 'r' for x in diff_df['r2_diff']])
        axes[1, 1].set_title(f'R² Differenz (Hybrid - Klassisch) (In={input_len}, Out={output_len})')
        axes[1, 1].set_ylabel('R² Differenz')
        axes[1, 1].axhline(y=0, color='black', linestyle='-')
        axes[1, 1].set_xticklabels(diff_df['asset'], rotation=45, ha='right')
        axes[1, 1].grid(True, axis='y')

        # DA Differenz
        axes[2, 0].bar(diff_df['asset'], diff_df['da_diff'], color=['g' if x > 0 else 'r' for x in diff_df['da_diff']])
        axes[2, 0].set_title(f'DA Differenz (Hybrid - Klassisch) (In={input_len}, Out={output_len})')
        axes[2, 0].set_ylabel('DA Differenz (%)')
        axes[2, 0].axhline(y=0, color='black', linestyle='-')
        axes[2, 0].set_xticklabels(diff_df['asset'], rotation=45, ha='right')
        axes[2, 0].grid(True, axis='y')

        # Leerer Plot für Symmetrie
        axes[2, 1].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_diff_by_asset_in{input_len}_out{output_len}.png')
        plt.close()

def run_multi_asset_tests():
    """Führt Tests für zwei verschiedene Input/Output-Konfigurationen über verschiedene Finanzinstrumente durch."""
    # Zwei Konfigurationen definieren
    configs = [
        (20, 5),   # Kurzfristige Prognose
        (60, 10)   # Mittelfristige Prognose
    ]

    # Zeitraum definieren (3-Jahres-Zeitraum)
    start_date = dt.datetime(2021, 1, 1)
    end_date = dt.datetime(2024, 1, 1)

    # Finanzinstrumente definieren
    assets = [
        {
            'name': 'S&P 500',
            'ticker': '^GSPC',
            'description': 'US-Aktienindex'
        },
        {
            'name': 'EURO STOXX 50',
            'ticker': '^STOXX50E',
            'description': 'Europäischer Aktienindex'
        },
        {
            'name': 'Gold',
            'ticker': 'GC=F',
            'description': 'Gold-Futures'
        },
        {
            'name': 'Bitcoin',
            'ticker': 'BTC-USD',
            'description': 'Kryptowährung'
        },
        {
            'name': 'VIX',
            'ticker': '^VIX',
            'description': 'Volatilitätsindex'
        }
    ]

    # Ergebnisse speichern
    results = []

    # Für jedes Asset und jede Konfiguration testen
    for asset in assets:
        print(f"\n===== Lade Daten für: {asset['name']} ({start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}) =====")

        try:
            # Daten laden
            data = yf.download(asset['ticker'], start=start_date, end=end_date)

            # Prüfen, ob genügend Daten vorhanden sind
            min_required = max([c[0] + c[1] + 100 for c in configs])  # Längste Konfiguration + Puffer
            if len(data) < min_required:
                print(f"Nicht genügend Daten für Asset {asset['name']}. Überspringe.")
                continue

            # Volatilität berechnen (für spätere Analyse)
            data['Returns'] = data['Close'].pct_change()
            asset_volatility = data['Returns'].std() * np.sqrt(252)  # Annualisierte Volatilität
            print(f"Annualisierte Volatilität für {asset['name']}: {asset_volatility:.2%}")

            # Für jede Konfiguration trainieren und evaluieren
            for input_len, output_len in configs:
                try:
                    asset_result = train_and_evaluate_for_asset(
                        data=data,
                        asset_name=asset['name'],
                        asset_volatility=asset_volatility,
                        input_len=input_len,
                        output_len=output_len
                    )
                    results.append(asset_result)
                except Exception as e:
                    print(f"Fehler bei Konfiguration Input={input_len}, Output={output_len} für {asset['name']}: {e}")

        except Exception as e:
            print(f"Fehler beim Laden der Daten für Asset {asset['name']}: {e}")
            continue

    # Ergebnisse in DataFrame umwandeln für bessere Analyse
    results_df = []

    for r in results:
        asset_name = r['asset_name']
        volatility = r['volatility']
        input_len = r['input_length']
        output_len = r['output_length']

        # Gesamtmetriken für klassisches Modell
        classic = {
            'model_type': 'Klassisch',
            'asset': asset_name,
            'volatility': volatility,
            'input_length': input_len,
            'output_length': output_len,
            'parameters': r['classic_model']['parameters'],
            'training_time': r['classic_model']['training_time'],
            'inference_time': r['classic_model']['metrics']['overall']['inference_time'],
            'mae': r['classic_model']['metrics']['overall']['mae'],
            'rmse': r['classic_model']['metrics']['overall']['rmse'],
            'mape': r['classic_model']['metrics']['overall']['mape'],
            'r2': r['classic_model']['metrics']['overall']['r2'],
            'da': r['classic_model']['metrics']['overall']['da']
        }

        # Gesamtmetriken für hybrides Modell
        hybrid = {
            'model_type': 'Hybrid',
            'asset': asset_name,
            'volatility': volatility,
            'input_length': input_len,
            'output_length': output_len,
            'parameters': r['hybrid_model']['parameters'],
            'training_time': r['hybrid_model']['training_time'],
            'inference_time': r['hybrid_model']['metrics']['overall']['inference_time'],
            'mae': r['hybrid_model']['metrics']['overall']['mae'],
            'rmse': r['hybrid_model']['metrics']['overall']['rmse'],
            'mape': r['hybrid_model']['metrics']['overall']['mape'],
            'r2': r['hybrid_model']['metrics']['overall']['r2'],
            'da': r['hybrid_model']['metrics']['overall']['da']
        }

        # Volatilitätsmetriken für klassisches Modell
        if 'high' in r['classic_model']['metrics']['volatility'] and 'r2' in r['classic_model']['metrics']['volatility']['high']:
            classic['high_vol_r2'] = r['classic_model']['metrics']['volatility']['high']['r2']
            classic['high_vol_da'] = r['classic_model']['metrics']['volatility']['high']['da']
            classic['high_vol_mae'] = r['classic_model']['metrics']['volatility']['high']['mae']
            classic['high_vol_mape'] = r['classic_model']['metrics']['volatility']['high']['mape']

        if 'normal' in r['classic_model']['metrics']['volatility'] and 'r2' in r['classic_model']['metrics']['volatility']['normal']:
            classic['normal_vol_r2'] = r['classic_model']['metrics']['volatility']['normal']['r2']
            classic['normal_vol_da'] = r['classic_model']['metrics']['volatility']['normal']['da']
            classic['normal_vol_mae'] = r['classic_model']['metrics']['volatility']['normal']['mae']
            classic['normal_vol_mape'] = r['classic_model']['metrics']['volatility']['normal']['mape']

        # Volatilitätsmetriken für hybrides Modell
        if 'high' in r['hybrid_model']['metrics']['volatility'] and 'r2' in r['hybrid_model']['metrics']['volatility']['high']:
            hybrid['high_vol_r2'] = r['hybrid_model']['metrics']['volatility']['high']['r2']
            hybrid['high_vol_da'] = r['hybrid_model']['metrics']['volatility']['high']['da']
            hybrid['high_vol_mae'] = r['hybrid_model']['metrics']['volatility']['high']['mae']
            hybrid['high_vol_mape'] = r['hybrid_model']['metrics']['volatility']['high']['mape']

        if 'normal' in r['hybrid_model']['metrics']['volatility'] and 'r2' in r['hybrid_model']['metrics']['volatility']['normal']:
            hybrid['normal_vol_r2'] = r['hybrid_model']['metrics']['volatility']['normal']['r2']
            hybrid['normal_vol_da'] = r['hybrid_model']['metrics']['volatility']['normal']['da']
            hybrid['normal_vol_mae'] = r['hybrid_model']['metrics']['volatility']['normal']['mae']
            hybrid['normal_vol_mape'] = r['hybrid_model']['metrics']['volatility']['normal']['mape']

        results_df.append(classic)
        results_df.append(hybrid)

    # In DataFrame konvertieren
    df_results = pd.DataFrame(results_df)

    # Ergebnisse speichern
    df_results.to_csv(f'{output_dir}/multi_asset_comparison_results.csv', index=False)

    # Ergebnisse visualisieren
    plot_multi_asset_results(df_results)

    return df_results

# Hauptausführung
if __name__ == "__main__":
    # Ausgabeverzeichnis erstellen, falls nicht vorhanden
    output_dir = 'output_results'
    os.makedirs(output_dir, exist_ok=True)

    results_df = run_multi_asset_tests()
    print(f"Multi-Asset-Tests abgeschlossen. Ergebnisse in '{output_dir}/multi_asset_comparison_results.csv'")
    print(results_df)
