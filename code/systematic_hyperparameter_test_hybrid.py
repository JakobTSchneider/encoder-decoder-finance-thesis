### Welche Parameter performen am besten?###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as dt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Activation, Dot, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
import os
import random
import time
import itertools

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

def create_sequences(data, input_len, output_len):
    """Erstellt Input- und Output-Sequenzen für Zeitreihenmodellierung."""
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

def build_hybrid_model(input_seq_length, output_seq_length, latent_dim=100, dropout_rate=0.1, regularization=0.0):
    """Erstellt ein hybrides Encoder-Decoder-Modell mit anpassbaren Hyperparametern."""
    # Encoder und Decoder Inputs
    encoder_inputs = Input(shape=(input_seq_length, 1))
    decoder_inputs = Input(shape=(output_seq_length, 1))

    # Encoder mit Bidirectional LSTM und Regularisierung
    encoder = Bidirectional(LSTM(latent_dim,
                                dropout=dropout_rate,
                                kernel_regularizer=l2(regularization),
                                recurrent_regularizer=l2(regularization),
                                return_sequences=True,
                                return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    # Zustände für den Decoder
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder mit Regularisierung
    decoder = LSTM(latent_dim*2,
                  dropout=dropout_rate,
                  kernel_regularizer=l2(regularization),
                  recurrent_regularizer=l2(regularization),
                  return_sequences=True,
                  return_state=True)
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
    predictions = model.predict([X_test, decoder_input_test], verbose=0)
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
        step_r2 = r2_score(step_actual_original, step_preds_original)

        horizon_metrics.append({
            'step': step + 1,
            'mae': step_mae,
            'rmse': step_rmse,
            'da': step_da,
            'r2': step_r2
        })

    # Ergebnisse zusammenfassen
    results = {
        'overall': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'da': da,
            'inference_time': inference_duration
        },
        'horizon': horizon_metrics
    }

    return results

def run_hyperparameter_search():
    """Führt eine systematische Hyperparameter-Suche für das hybride Modell durch."""
    print("Loading S&P 500 data...")
    end_date = dt.datetime(2024, 4, 1)
    start_date = dt.datetime(2018, 4, 1)
    data = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

    # Datenvorverarbeitung
    df = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Zu testende Konfigurationen
    input_lengths = [20, 60, 90]  # Verschiedene Input-Sequenzlängen
    output_lengths = [5, 10, 20]  # Verschiedene Output-Sequenzlängen
    latent_dims = [50, 100, 150, 200]  # Verschiedene latent dimensions
    dropout_rates = [0.1, 0.2, 0.3, 0.4]  # Verschiedene Dropout-Raten
    regularization_values = [0.0]  # Keine Regularisierung für einfachere Suche

    # Ergebnisse speichern
    results = []

    # Konfigurationen erzeugen
    configs = list(itertools.product(input_lengths, output_lengths, latent_dims, dropout_rates, regularization_values))
    print(f"Insgesamt {len(configs)} Konfigurationen zu testen.")

    # Konfigurationszähler für Fortschrittsanzeige
    total_configs = len(configs)
    current_config = 0

    for input_len, output_len, latent_dim, dropout_rate, regularization in configs:
        current_config += 1
        # Fortschritt anzeigen
        print(f"Konfiguration {current_config}/{total_configs} ({current_config/total_configs*100:.1f}%)")
        print(f"Test: Input={input_len}, Output={output_len}, LD={latent_dim}, DR={dropout_rate}")

        # Daten vorbereiten (einmalig für jede Input/Output-Konfiguration)
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
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0001
        )

        time_callback = TimeHistory()

        # Modell mit aktuellen Hyperparametern bauen
        model = build_hybrid_model(
            input_len,
            output_len,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            regularization=regularization
        )

        # Parameterzahl ermitteln
        params_count = model.count_params()

        # Modell trainieren
        try:
            history = model.fit(
                [X_train, decoder_input_train],
                y_train,
                validation_data=([X_val, decoder_input_val], y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, time_callback],
                verbose=0,
                shuffle=False
            )

            training_time = time_callback.total_training_time

            # Modell evaluieren
            eval_results = evaluate_model(
                model, X_test, decoder_input_test, y_test, scaler, output_len
            )

            # Validierungsverlauf extrahieren
            val_loss = history.history['val_loss']
            epochs_completed = len(val_loss)
            best_val_loss = min(val_loss)

            # Ergebnisse speichern
            result = {
                'input_length': input_len,
                'output_length': output_len,
                'latent_dim': latent_dim,
                'dropout_rate': dropout_rate,
                'regularization': regularization,
                'parameters': params_count,
                'training_time': training_time,
                'epochs_completed': epochs_completed,
                'best_val_loss': best_val_loss,
                'mae': eval_results['overall']['mae'],
                'rmse': eval_results['overall']['rmse'],
                'r2': eval_results['overall']['r2'],
                'da': eval_results['overall']['da'],
                'inference_time': eval_results['overall']['inference_time']
            }

            results.append(result)

            # Kurze Zusammenfassung nach jedem Test ausgeben
            one_day_mae = eval_results['horizon'][0]['mae']
            print(f"Metriken: MAE={eval_results['overall']['mae']:.2f}, MAE (1-Tag)={one_day_mae:.2f}, DA={eval_results['overall']['da']:.2f}%, R²={eval_results['overall']['r2']:.4f}")
            print(f"Training: {epochs_completed} Epochen, {training_time:.2f} Sekunden\n")

        except Exception as e:
            print(f"Fehler bei Konfiguration: IN={input_len}, OUT={output_len}, LD={latent_dim}, DR={dropout_rate}")
            print(f"Fehler: {e}\n")

        # Speicherplatz freigeben
        tf.keras.backend.clear_session()

        # Ergebnisse zwischenspeichern nach jeder Konfiguration
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{output_dir}/hybrid_hyperparameter_search.csv', index=False)

    # Finale Ergebnisse
    final_results_df = pd.DataFrame(results)
    final_results_df.to_csv(f'{output_dir}/hybrid_hyperparameter_search_final.csv', index=False)

    # Visualisierungen erstellen
    create_hyperparameter_visualizations(final_results_df)

    return final_results_df

def create_hyperparameter_visualizations(df):
    """Erstellt Visualisierungen für die Hyperparameter-Suche."""
    # Heatmap für verschiedene Kombinationen von latent_dim und dropout_rate
    for input_len in df['input_length'].unique():
        for output_len in df['output_length'].unique():
            subset = df[(df['input_length'] == input_len) &
                         (df['output_length'] == output_len)]

            if len(subset) > 0:
                # MAE-Heatmap
                plt.figure(figsize=(10, 8))
                pivot_mae = subset.pivot_table(
                    index='dropout_rate',
                    columns='latent_dim',
                    values='mae'
                )
                sns.heatmap(pivot_mae, annot=True, cmap='YlGnBu_r', fmt='.1f')
                plt.title(f'MAE: Input={input_len}, Output={output_len}')
                plt.xlabel('Latent Dimension')
                plt.ylabel('Dropout Rate')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_mae_in{input_len}_out{output_len}.png')
                plt.close()

                # R²-Heatmap
                plt.figure(figsize=(10, 8))
                pivot_r2 = subset.pivot_table(
                    index='dropout_rate',
                    columns='latent_dim',
                    values='r2'
                )
                sns.heatmap(pivot_r2, annot=True, cmap='YlGnBu', fmt='.4f')
                plt.title(f'R²: Input={input_len}, Output={output_len}')
                plt.xlabel('Latent Dimension')
                plt.ylabel('Dropout Rate')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_r2_in{input_len}_out{output_len}.png')
                plt.close()

                # DA-Heatmap
                plt.figure(figsize=(10, 8))
                pivot_da = subset.pivot_table(
                    index='dropout_rate',
                    columns='latent_dim',
                    values='da'
                )
                sns.heatmap(pivot_da, annot=True, cmap='YlGnBu', fmt='.1f')
                plt.title(f'Direktionale Genauigkeit: Input={input_len}, Output={output_len}')
                plt.xlabel('Latent Dimension')
                plt.ylabel('Dropout Rate')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_da_in{input_len}_out{output_len}.png')
                plt.close()

    # Übergreifende Analyse über alle Input/Output-Längen
    plt.figure(figsize=(12, 10))

    # Gruppierung nach Latent Dimension und Dropout
    grouped_by_hyperparams = df.groupby(['latent_dim', 'dropout_rate']).agg({
        'mae': 'mean',
        'rmse': 'mean',
        'r2': 'mean',
        'da': 'mean',
        'training_time': 'mean'
    }).reset_index()

    # MAE nach Hyperparametern
    plt.subplot(2, 2, 1)
    pivot_mae_all = grouped_by_hyperparams.pivot_table(
        index='dropout_rate',
        columns='latent_dim',
        values='mae'
    )
    sns.heatmap(pivot_mae_all, annot=True, cmap='YlGnBu_r', fmt='.1f')
    plt.title('Durchschnittliche MAE nach Hyperparametern')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Dropout Rate')

    # R² nach Hyperparametern
    plt.subplot(2, 2, 2)
    pivot_r2_all = grouped_by_hyperparams.pivot_table(
        index='dropout_rate',
        columns='latent_dim',
        values='r2'
    )
    sns.heatmap(pivot_r2_all, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('Durchschnittliches R² nach Hyperparametern')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Dropout Rate')

    # DA nach Hyperparametern
    plt.subplot(2, 2, 3)
    pivot_da_all = grouped_by_hyperparams.pivot_table(
        index='dropout_rate',
        columns='latent_dim',
        values='da'
    )
    sns.heatmap(pivot_da_all, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Durchschnittliche DA nach Hyperparametern')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Dropout Rate')

    # Trainingszeit nach Hyperparametern
    plt.subplot(2, 2, 4)
    pivot_time_all = grouped_by_hyperparams.pivot_table(
        index='dropout_rate',
        columns='latent_dim',
        values='training_time'
    )
    sns.heatmap(pivot_time_all, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title('Durchschnittliche Trainingszeit (s) nach Hyperparametern')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Dropout Rate')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_summary.png')
    plt.close()

    # Top-10 Konfigurationen nach verschiedenen Metriken
    print("\nTop-10 Konfigurationen nach MAE:")
    top_mae = df.sort_values('mae').head(10)[['input_length', 'output_length', 'latent_dim',
                                              'dropout_rate', 'mae', 'da', 'r2']]
    print(top_mae)

    print("\nTop-10 Konfigurationen nach R²:")
    top_r2 = df.sort_values('r2', ascending=False).head(10)[['input_length', 'output_length', 'latent_dim',
                                                           'dropout_rate', 'mae', 'da', 'r2']]
    print(top_r2)

    print("\nTop-10 Konfigurationen nach DA:")
    top_da = df.sort_values('da', ascending=False).head(10)[['input_length', 'output_length', 'latent_dim',
                                                           'dropout_rate', 'mae', 'da', 'r2']]
    print(top_da)

    # Rangkorrelation zwischen Hyperparametern und Metriken
    correlation = df[['latent_dim', 'dropout_rate', 'mae', 'rmse', 'r2', 'da']].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Spearman-Rangkorrelation zwischen Hyperparametern und Metriken')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_correlation.png')
    plt.close()

# Hauptausführung
if __name__ == "__main__":
    print("Starte systematische Hyperparameter-Suche für das hybride Modell...")
    try:
        results_df = run_hyperparameter_search()
        print("\nHyperparameter-Suche abgeschlossen!")
        print(f"Ergebnisse wurden in '{output_dir}/hybrid_hyperparameter_search_final.csv' gespeichert.")
        print("Visualisierungen wurden als PNG-Dateien gespeichert.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
