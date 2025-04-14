### Test mit Latent_dim =100 und Dropout = 0,1 bei hybrid, Input/Output: 20/5, 40/5, 40/10, 60/10, 60/20, 90/20 ###

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
            # R² für hohe Volatilität hinzugefügt
            high_vol_metrics['r2'] = r2_score(high_vol_actual, high_vol_pred)

    if len(low_vol_actual) > 0:
        low_vol_metrics['mae'] = mean_absolute_error(low_vol_actual, low_vol_pred)
        low_vol_metrics['rmse'] = np.sqrt(mean_squared_error(low_vol_actual, low_vol_pred))
        if len(low_vol_actual) > 1:
            low_vol_metrics['da'] = directional_accuracy(low_vol_actual, low_vol_pred)
            # R² für niedrige Volatilität hinzugefügt
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
        step_da = directional_accuracy(step_actual_original, step_preds_original)
        # R² für jeden Prognosehorizont hinzugefügt
        step_r2 = r2_score(step_actual_original, step_preds_original)

        horizon_metrics.append({
            'step': step + 1,
            'mae': step_mae,
            'rmse': step_rmse,
            'da': step_da,
            'r2': step_r2  # R² für jeden Schritt hinzugefügt
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

# Hauptprogramm für die systematischen Tests
def run_systematic_tests():
    # Daten laden
    print("Lade S&P 500 Daten...")
    end_date = dt.datetime(2024, 4, 1)
    start_date = dt.datetime(2018, 4, 1)
    data = yf.download("^GSPC", start=start_date, end=end_date)

    # Datenvorverarbeitung
    df = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Zu testende Konfigurationen - fokussierte Auswahl
    test_configs = [
        (20, 5), (40, 5),    # Kurzfristig
        (40, 10), (60, 10),  # Mittelfristig
        (60, 20), (90, 20)   # Längerfristig
    ]

    # Ergebnisse speichern
    results = []

    # Für jede Kombination durchführen
    for input_len, output_len in test_configs:
        print(f"\n===== Test: Input={input_len}, Output={output_len} =====")

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
        classic_model.summary()
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
        hybrid_model.summary()
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
        config_results = {
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

        results.append(config_results)

        # R²-Vergleich visualisieren
        plt.figure(figsize=(12, 6))
        classic_r2 = [m['r2'] for m in classic_results['horizon']]
        hybrid_r2 = [m['r2'] for m in hybrid_results['horizon']]

        x = range(1, output_len + 1)
        plt.plot(x, classic_r2, 'b-o', label='Klassisches Modell')
        plt.plot(x, hybrid_r2, 'r-o', label='Hybrides Modell')

        plt.title(f'R² Bestimmtheitsmaß: Input={input_len}, Output={output_len}')
        plt.xlabel('Prognosehorizont (Tage)')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        plt.ylim(min(min(classic_r2), min(hybrid_r2))-0.1, 1.0)  # Angepasster y-Bereich für R²
        plt.savefig(f'{output_dir}/r2_comparison_in{input_len}_out{output_len}.png')
        plt.close()

        # Direktionale Genauigkeitsvergleich visualisieren
        plt.figure(figsize=(12, 6))
        classic_da = [m['da'] for m in classic_results['horizon']]
        hybrid_da = [m['da'] for m in hybrid_results['horizon']]

        x = range(1, output_len + 1)
        plt.plot(x, classic_da, 'b-o', label='Klassisches Modell')
        plt.plot(x, hybrid_da, 'r-o', label='Hybrides Modell')
        plt.axhline(y=50, color='gray', linestyle='--', label='Zufallsniveau (50%)')

        plt.title(f'Direktionale Genauigkeit: Input={input_len}, Output={output_len}')
        plt.xlabel('Prognosehorizont (Tage)')
        plt.ylabel('Direktionale Genauigkeit (%)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        plt.savefig(f'{output_dir}/da_comparison_in{input_len}_out{output_len}.png')
        plt.close()

        # MAE-Vergleich visualisieren
        plt.figure(figsize=(12, 6))
        classic_mae = [m['mae'] for m in classic_results['horizon']]
        hybrid_mae = [m['mae'] for m in hybrid_results['horizon']]

        plt.plot(x, classic_mae, 'b-o', label='Klassisches Modell')
        plt.plot(x, hybrid_mae, 'r-o', label='Hybrides Modell')

        plt.title(f'MAE: Input={input_len}, Output={output_len}')
        plt.xlabel('Prognosehorizont (Tage)')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/mae_comparison_in{input_len}_out{output_len}.png')
        plt.close()

        # Speicherplatz freigeben
        tf.keras.backend.clear_session()

    # Ergebnisse in DataFrame umwandeln für bessere Analyse
    results_df = []

    for r in results:
        input_len = r['input_length']
        output_len = r['output_length']

        # Gesamtmetriken für klassisches Modell
        classic = {
            'model_type': 'Klassisch',
            'input_length': input_len,
            'output_length': output_len,
            'parameters': r['classic_model']['parameters'],
            'training_time': r['classic_model']['training_time'],
            'inference_time': r['classic_model']['metrics']['overall']['inference_time'],
            'mae': r['classic_model']['metrics']['overall']['mae'],
            'rmse': r['classic_model']['metrics']['overall']['rmse'],
            'r2': r['classic_model']['metrics']['overall']['r2'],  # R² in Ergebnisse aufnehmen
            'da': r['classic_model']['metrics']['overall']['da']
        }

        # Gesamtmetriken für hybrides Modell
        hybrid = {
            'model_type': 'Hybrid',
            'input_length': input_len,
            'output_length': output_len,
            'parameters': r['hybrid_model']['parameters'],
            'training_time': r['hybrid_model']['training_time'],
            'inference_time': r['hybrid_model']['metrics']['overall']['inference_time'],
            'mae': r['hybrid_model']['metrics']['overall']['mae'],
            'rmse': r['hybrid_model']['metrics']['overall']['rmse'],
            'r2': r['hybrid_model']['metrics']['overall']['r2'],  # R² in Ergebnisse aufnehmen
            'da': r['hybrid_model']['metrics']['overall']['da']
        }

        results_df.append(classic)
        results_df.append(hybrid)

    # In DataFrame konvertieren
    df_results = pd.DataFrame(results_df)

    # Ergebnisse speichern
    df_results.to_csv(f'{output_dir}/model_comparison_results.csv', index=False)

    # Ergebnisse visualisieren
    plot_summary_results(df_results)

    return df_results

def plot_summary_results(df):
    """Erstellt Zusammenfassungsgrafiken für die Ergebnisse."""
    # Input-Länge vs. MAE nach Modelltyp und Output-Länge
    plt.figure(figsize=(15, 10))

    for output_len in df['output_length'].unique():
        plt.subplot(len(df['output_length'].unique()), 1, list(df['output_length'].unique()).index(output_len) + 1)

        subset = df[df['output_length'] == output_len]
        classic = subset[subset['model_type'] == 'Klassisch']
        hybrid = subset[subset['model_type'] == 'Hybrid']

        plt.plot(classic['input_length'], classic['mae'], 'bo-', label='Klassisch')
        plt.plot(hybrid['input_length'], hybrid['mae'], 'ro-', label='Hybrid')

        plt.title(f'MAE vs. Input-Länge (Output-Länge = {output_len})')
        plt.xlabel('Input-Länge')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/input_length_vs_mae.png')
    plt.close()

    # R²-Vergleich hinzugefügt
    plt.figure(figsize=(15, 10))

    for output_len in df['output_length'].unique():
        plt.subplot(len(df['output_length'].unique()), 1, list(df['output_length'].unique()).index(output_len) + 1)

        subset = df[df['output_length'] == output_len]
        classic = subset[subset['model_type'] == 'Klassisch']
        hybrid = subset[subset['model_type'] == 'Hybrid']

        plt.plot(classic['input_length'], classic['r2'], 'bo-', label='Klassisch')
        plt.plot(hybrid['input_length'], hybrid['r2'], 'ro-', label='Hybrid')

        plt.title(f'R² vs. Input-Länge (Output-Länge = {output_len})')
        plt.xlabel('Input-Länge')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/input_length_vs_r2.png')
    plt.close()

    # Direktionale Genauigkeit
    plt.figure(figsize=(15, 10))

    for output_len in df['output_length'].unique():
        plt.subplot(len(df['output_length'].unique()), 1, list(df['output_length'].unique()).index(output_len) + 1)

        subset = df[df['output_length'] == output_len]
        classic = subset[subset['model_type'] == 'Klassisch']
        hybrid = subset[subset['model_type'] == 'Hybrid']

        plt.plot(classic['input_length'], classic['da'], 'bo-', label='Klassisch')
        plt.plot(hybrid['input_length'], hybrid['da'], 'ro-', label='Hybrid')

        plt.axhline(y=50, color='gray', linestyle='--', label='Zufallsniveau (50%)')
        plt.title(f'Direktionale Genauigkeit vs. Input-Länge (Output-Länge = {output_len})')
        plt.xlabel('Input-Länge')
        plt.ylabel('Direktionale Genauigkeit (%)')
        plt.legend()
        plt.grid(True)
        plt.ylim(40, 100)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/input_length_vs_da.png')
    plt.close()

    # Trainingszeit-Vergleich
    plt.figure(figsize=(15, 10))

    for output_len in df['output_length'].unique():
        plt.subplot(len(df['output_length'].unique()), 1, list(df['output_length'].unique()).index(output_len) + 1)

        subset = df[df['output_length'] == output_len]
        classic = subset[subset['model_type'] == 'Klassisch']
        hybrid = subset[subset['model_type'] == 'Hybrid']

        plt.plot(classic['input_length'], classic['training_time'], 'bo-', label='Klassisch')
        plt.plot(hybrid['input_length'], hybrid['training_time'], 'ro-', label='Hybrid')

        plt.title(f'Trainingszeit vs. Input-Länge (Output-Länge = {output_len})')
        plt.xlabel('Input-Länge')
        plt.ylabel('Trainingszeit (s)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/input_length_vs_training_time.png')
    plt.close()

    # Heatmap der MAE-Differenz (Klassisch - Hybrid)
    plt.figure(figsize=(10, 8))

    # Pivot-Tabelle erstellen
    pivot_df = df.pivot_table(
        index='input_length',
        columns=['output_length', 'model_type'],
        values='mae'
    )

    # Differenz berechnen (Klassisch - Hybrid)
    diff_df = pd.DataFrame(index=pivot_df.index)

    for output_len in df['output_length'].unique():
        diff_df[output_len] = pivot_df[(output_len, 'Klassisch')] - pivot_df[(output_len, 'Hybrid')]

    # Heatmap
    sns.heatmap(diff_df, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
    plt.title('MAE-Differenz: Klassisch - Hybrid\n(Positiv = Klassisch schlechter, Negativ = Hybrid schlechter)')
    plt.xlabel('Output-Länge')
    plt.ylabel('Input-Länge')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mae_diff_heatmap.png')
    plt.close()

    # Heatmap der R²-Differenz (Hybrid - Klassisch) hinzugefügt
    plt.figure(figsize=(10, 8))

    # Pivot-Tabelle erstellen
    pivot_df = df.pivot_table(
        index='input_length',
        columns=['output_length', 'model_type'],
        values='r2'
    )

    # Differenz berechnen (Hybrid - Klassisch) für R² (hier ist höher besser)
    diff_df = pd.DataFrame(index=pivot_df.index)

    for output_len in df['output_length'].unique():
        diff_df[output_len] = pivot_df[(output_len, 'Hybrid')] - pivot_df[(output_len, 'Klassisch')]

    # Heatmap
    sns.heatmap(diff_df, annot=True, cmap='RdBu', center=0, fmt='.4f')
    plt.title('R²-Differenz: Hybrid - Klassisch\n(Positiv = Hybrid besser, Negativ = Klassisch besser)')
    plt.xlabel('Output-Länge')
    plt.ylabel('Input-Länge')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_diff_heatmap.png')
    plt.close()

    # Gleiche Heatmap für Direktionale Genauigkeit
    plt.figure(figsize=(10, 8))

    # Pivot-Tabelle erstellen
    pivot_df = df.pivot_table(
        index='input_length',
        columns=['output_length', 'model_type'],
        values='da'
    )

    # Differenz berechnen (Hybrid - Klassisch) für DA (hier ist höher besser)
    diff_df = pd.DataFrame(index=pivot_df.index)

    for output_len in df['output_length'].unique():
        diff_df[output_len] = pivot_df[(output_len, 'Hybrid')] - pivot_df[(output_len, 'Klassisch')]

    # Heatmap
    sns.heatmap(diff_df, annot=True, cmap='RdBu', center=0, fmt='.4f')
    plt.title('DA-Differenz: Hybrid - Klassisch\n(Positiv = Hybrid besser, Negativ = Klassisch besser)')
    plt.xlabel('Output-Länge')
    plt.ylabel('Input-Länge')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/da_diff_heatmap.png')
    plt.close()

# Hauptausführung
if __name__ == "__main__":
    results_df = run_systematic_tests()
    print(f"Systematischer Test abgeschlossen. Ergebnisse in '{output_dir}/model_comparison_results.csv'")
    print(results_df)
