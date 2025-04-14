# Block 2: Klassisches Encoder-Decoder Modell

# ----- 1. Modell definieren (Multi-Step) -----
print("Erstelle klassisches Multi-Step-Modell...")

# Definieren der Encoder- und Decoder-Eingaben
encoder_inputs = Input(shape=(input_seq_length, 1))
decoder_inputs = Input(shape=(output_seq_length, 1))

# Encoder
encoder_lstm = LSTM(180, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_lstm = LSTM(180, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Ausgabeschicht für Multi-Step
output = TimeDistributed(Dense(1))(decoder_outputs)

# Modell definieren
classic_model = Model([encoder_inputs, decoder_inputs], output)
classic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Modellzusammenfassung anzeigen
classic_model.summary()
classic_params = classic_model.count_params()
print(f"Gesamtanzahl der Parameter: {classic_params:,}")

# ----- 2. Modell Training -----
print("Starte Training...")

# Early Stopping und Zeitmessung
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

time_callback = TimeHistory()

# Start-Zeitpunkt für die Zeitmessung
training_start_time = time.time()

# Training
history = classic_model.fit(
    [X_train, decoder_input_train],
    y_train,
    validation_data=([X_val, decoder_input_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, time_callback],
    verbose=1,
    shuffle=False #sinnvoll für Zeitreihen
)

# End-Zeitpunkt für die Zeitmessung
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Gesamte Trainingszeit: {training_duration:.2f} Sekunden")
print(f"Durchschnittliche Zeit pro Epoche: {np.mean(time_callback.epoch_times):.2f} Sekunden")

# ----- 3. Evaluierung -----
print("Evaluiere Modell...")

# Zeitmessung für die Inferenz
inference_start_time = time.time()
predictions = classic_model.predict([X_test, decoder_input_test])
inference_end_time = time.time()
inference_duration = inference_end_time - inference_start_time
print(f"Inferenzzeit für {len(X_test)} Testfälle: {inference_duration:.2f} Sekunden")
print(f"Durchschnittliche Inferenzzeit pro Testfall: {inference_duration/len(X_test)*1000:.2f} ms")

# Rücktransformation für Metriken
predictions_flat = predictions.reshape(-1, 1)
y_test_flat = y_test.reshape(-1, 1)

# Zurück in die ursprüngliche Skala
predictions_original = scaler.inverse_transform(predictions_flat)
y_test_original = scaler.inverse_transform(y_test_flat)

# Neuformatierung für zeitschrittweise Metriken
predictions_by_step = []
actual_by_step = []

for step in range(output_seq_length):
    step_preds = predictions[:, step, 0].reshape(-1, 1)
    step_actual = y_test[:, step, 0].reshape(-1, 1)

    # Zurück in die ursprüngliche Skala
    step_preds_original = scaler.inverse_transform(step_preds)
    step_actual_original = scaler.inverse_transform(step_actual)

    predictions_by_step.append(step_preds_original)
    actual_by_step.append(step_actual_original)

# Gesamtperformance
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

# Direktionale Genauigkeit berechnen
da = directional_accuracy(y_test_original, predictions_original)

# SMAPE berechnen
smape = 100 * np.mean(2 * np.abs(predictions_original - y_test_original) /
                     (np.abs(predictions_original) + np.abs(y_test_original)))

print("\nGesamtmodell-Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"SMAPE: {smape:.4f}%")
print(f"R²: {r2:.4f}")
print(f"Direktionale Genauigkeit: {da:.2f}%")

# Performance pro Zeitschritt
print("\nPerformance nach Prognosehorizont:")
for step in range(output_seq_length):
    step_mae = mean_absolute_error(actual_by_step[step], predictions_by_step[step])
    step_rmse = np.sqrt(mean_squared_error(actual_by_step[step], predictions_by_step[step]))

    # Direktionale Genauigkeit für den einzelnen Schritt
    step_da = directional_accuracy(actual_by_step[step], predictions_by_step[step])

    # SMAPE für den einzelnen Schritt
    step_smape = 100 * np.mean(2 * np.abs(predictions_by_step[step] - actual_by_step[step]) /
                           (np.abs(predictions_by_step[step]) + np.abs(actual_by_step[step])))

    print(f"Tag {step+1}:")
    print(f"  MAE: {step_mae:.4f}")
    print(f"  RMSE: {step_rmse:.4f}")
    print(f"  SMAPE: {step_smape:.4f}%")
    print(f"  Direktionale Genauigkeit: {step_da:.2f}%")

# Die Datumsindizes für die Testdaten ermitteln
total_seq_count = len(X)
test_start_idx = train_size + val_size
seq_offset = test_start_idx + input_seq_length  # Offset zum ursprünglichen Datensatz

# Extrahieren der entsprechenden Datumsindizes
test_start_date = df.index[seq_offset]
test_dates = df.index[seq_offset:seq_offset + len(y_test)]

# Analyse nach Volatilitätsphasen
print("\nAnalyse nach Volatilitätsphasen:")
# Wir verwenden nur die Vorhersagen für den ersten Tag für die Volatilitätsanalyse
first_day_predictions = predictions_by_step[0]
first_day_actual = actual_by_step[0]

# Volatilitätsanalyse durchführen
volatility_threshold = 0.01  # 1% tägliche Preisänderung als Schwellenwert
high_vol_metrics, low_vol_metrics = analyze_by_volatility(
    first_day_actual,
    first_day_predictions,
    volatility_threshold
)

print("Hochvolatilitätsphasen:")
for metric, value in high_vol_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

print("Normalmarktphasen:")
for metric, value in low_vol_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

# ----- 4. Visualisierung -----
# Trainings- und Validierungsverlust plotten
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Trainingsverlust')
plt.plot(history.history['val_loss'], label='Validierungsverlust')
plt.title('Modellverlust (Klassisches Modell)')
plt.ylabel('Verlust')
plt.xlabel('Epoche')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('classic_model_loss.png')
print("Verlaufsgrafik gespeichert als 'classic_model_loss.png'")

# Direktionale Genauigkeit nach Prognosehorizont
plt.figure(figsize=(10, 6))
days = np.arange(1, output_seq_length + 1)
directional_accuracies = []

for step in range(output_seq_length):
    step_actual = actual_by_step[step].flatten()
    step_predicted = predictions_by_step[step].flatten()
    step_da = directional_accuracy(actual_by_step[step], predictions_by_step[step])
    directional_accuracies.append(step_da)

plt.bar(days, directional_accuracies, color='skyblue')
plt.axhline(y=50, color='r', linestyle='--', label='Zufallsniveau (50%)')
plt.title('Direktionale Genauigkeit nach Prognosehorizont (Klassisches Modell)')
plt.xlabel('Prognosetag')
plt.ylabel('Direktionale Genauigkeit (%)')
plt.ylim(0, 100)
plt.xticks(days)
plt.legend()
plt.grid(True, axis='y')
plt.savefig('classic_directional_accuracy.png')
print("Direktionale Genauigkeitsgrafik gespeichert als 'classic_directional_accuracy.png'")

# Zusammenfassung
print("\n----- Modellzusammenfassung -----")
print(f"Modelltyp: Klassisches Encoder-Decoder-Modell")
print(f"Parameter: {classic_params:,}")
print(f"Trainingszeit: {training_duration:.2f} Sekunden")
print(f"Inferenzzeit für {len(X_test)} Testfälle: {inference_duration:.2f} Sekunden")
print("\nModellperformance:")
print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.4f}%")
print(f"  R²: {r2:.4f}, Direktionale Genauigkeit: {da:.2f}%")
