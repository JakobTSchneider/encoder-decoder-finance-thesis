# Block 3: Hybrides Encoder-Decoder Modell

# ----- 1. Modell definieren -----
print("Erstelle hybrides Encoder-Decoder-Modell...")
latent_dim = 100
dropout_rate = 0.1

# Eingabe-Layer
encoder_inputs = Input(shape=(input_seq_length, 1), name='encoder_inputs')
decoder_inputs = Input(shape=(output_seq_length, 1), name='decoder_inputs')

# Encoder mit Bidirectional LSTM
encoder = Bidirectional(LSTM(latent_dim, dropout=dropout_rate, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

# Zustände für den Decoder kombinieren
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

# Modell zusammensetzen
hybrid_model = Model([encoder_inputs, decoder_inputs], output)
hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Modellinfo
hybrid_model.summary()
hybrid_params = hybrid_model.count_params()
print(f"Gesamtanzahl der Parameter: {hybrid_params:,}")

# ----- 2. Training -----
print("Starte Training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
time_callback = TimeHistory()

# Training starten
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

# Training auswerten
hybrid_training_duration = time_callback.total_training_time
print(f"Trainingszeit: {hybrid_training_duration:.2f} Sekunden")
print(f"Zeit pro Epoche: {np.mean(time_callback.epoch_times):.2f} Sekunden")

# ----- 3. Evaluierung -----
print("Evaluiere hybrides Modell...")
inference_start_time = time.time()
hybrid_predictions = hybrid_model.predict([X_test, decoder_input_test])
inference_duration = time.time() - inference_start_time
print(f"Inferenzzeit für {len(X_test)} Testfälle: {inference_duration:.2f} Sekunden")
print(f"Inferenzzeit pro Testfall: {inference_duration/len(X_test)*1000:.2f} ms")

# Rücktransformation für Metriken
predictions_flat = hybrid_predictions.reshape(-1, 1)
y_test_flat = y_test.reshape(-1, 1)
predictions_original = scaler.inverse_transform(predictions_flat)
y_test_original = scaler.inverse_transform(y_test_flat)

# Gesamtmetriken
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)
da = directional_accuracy(y_test_original, predictions_original)
smape = 100 * np.mean(2 * np.abs(predictions_original - y_test_original) /
                     (np.abs(predictions_original) + np.abs(y_test_original)))

print("\nGesamtmodell-Performance (Hybrid):")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"SMAPE: {smape:.4f}%")
print(f"R²: {r2:.4f}")
print(f"Direktionale Genauigkeit: {da:.2f}%")

# Performance nach Prognosehorizont
print("\nPerformance nach Prognosehorizont (Hybrid):")
hybrid_predictions_by_step, hybrid_actual_by_step = evaluate_by_horizon(hybrid_predictions, y_test, scaler, output_seq_length)

# Analyse nach Volatilitätsphasen
print("\nAnalyse nach Volatilitätsphasen (Hybrid):")
# Volatilitätsanalyse für ersten Tag
first_day_predictions = hybrid_predictions_by_step[0]
first_day_actual = hybrid_actual_by_step[0]
high_vol_metrics, low_vol_metrics = analyze_by_volatility(
    first_day_actual,
    first_day_predictions,
    0.01
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
plt.plot(hybrid_history.history['loss'], label='Trainingsverlust')
plt.plot(hybrid_history.history['val_loss'], label='Validierungsverlust')
plt.title('Modellverlust (Hybrides Modell)')
plt.ylabel('Verlust')
plt.xlabel('Epoche')
plt.legend()
plt.grid(True)
plt.savefig('hybrid_model_loss.png')

# Direktionale Genauigkeit nach Prognosehorizont
plt.figure(figsize=(10, 6))
days = np.arange(1, output_seq_length + 1)
directional_accuracies = [directional_accuracy(hybrid_actual_by_step[step], hybrid_predictions_by_step[step])
                          for step in range(output_seq_length)]

plt.bar(days, directional_accuracies, color='skyblue')
plt.axhline(y=50, color='r', linestyle='--', label='Zufallsniveau (50%)')
plt.title('Direktionale Genauigkeit nach Prognosehorizont (Hybrides Modell)')
plt.xlabel('Prognosetag')
plt.ylabel('Direktionale Genauigkeit (%)')
plt.ylim(0, 100)
plt.xticks(days)
plt.legend()
plt.grid(True, axis='y')
plt.savefig('hybrid_directional_accuracy.png')

# Zusammenfassung
print("\n----- Hybrides Modell Zusammenfassung -----")
print(f"Modelltyp: Hybrides Encoder-Decoder-Modell mit Bidirectional LSTM und Attention")
print(f"Parameter: {hybrid_params:,}")
print(f"Trainingszeit: {hybrid_training_duration:.2f} Sekunden")
print(f"Inferenzzeit für {len(X_test)} Testfälle: {inference_duration:.2f} Sekunden")
print("\nModellperformance:")
print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.4f}%")
print(f"  R²: {r2:.4f}, Direktionale Genauigkeit: {da:.2f}%")
