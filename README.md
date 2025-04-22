# Encoder-Decoder Models for Financial Time Series Forecasting

This repository contains the complete code, data preparation, and evaluation framework developed for a Bachelor's thesis focusing on the application of encoder-decoder architectures to financial time series forecasting.

---

## 📘 Academic Context

This work supports a Bachelor's thesis in economics and data science. The project systematically compares the robustness and generalizability of two different encoder-decoder models (one conventional and one advanceed) for forecasting financial time series.


---
📌 Key Insights

- The classic encoder-decoder model performed more robustly across different input-output combinations, time periods and asset classes.
- The hybrid model performed well in volatile market phases and showed a higher tendency to predict the correct directional movement, but was more prone to overfitting.
- Results highlight trade-offs between model complexity and generalizability.

---

## 🧱 Model Architecture

The project is structured into several logical blocks, similar to a modular notebook:

### 🔹 Block 1: Imports, Seeds and Configuration
- All relevant libraries (TensorFlow, NumPy, pandas, etc.)
- Reproducibility settings for GPU and CPU
- Helper functions for metrics and volatility segmentation
- Dataset loading (S&P 500 via `yfinance`)
- Sequence creation and data splitting

➡️ Code: `code/01_preprocessing.py`

---

### 🔹 Block 2: Classic Encoder-Decoder Model
- LSTM-based sequence-to-sequence model
- Teacher forcing decoder
- Multi-step forecast (t+1 to t+20)
- Early stopping and training callbacks

➡️ Code: `code/02_model_classic.py` 

---

### 🔹 Block 3: Hybrid Encoder-Decoder Model
- Bidirectional LSTM encoder
- Dot-product attention mechanism
- Autoregressive decoder structure
- Comparison with classical model

➡️ Code: `code/03_model_hybrid.py` 

---

## ⚙️ Automated Experiments

A suite of automated tests evaluates both models under varying conditions:

### 🔬 Hyperparameter Search (Hybrid Model)
- Dropout rates, latent dimensions, input/output lengths
- Automated heatmap generation and CSV export

➡️ Code: `code/shared/systematic_hyperparameter_test_hybrid.py` 

### 🧪 Testing Framework
Both models are tested under different experimental setups:

- **Input/Output length variations** (e.g. 20/5, 60/20, etc.)
- ➡️ Code: `code/shared/11_input-output-configurations.py` 
- **Time period robustness** (e.g. crisis vs. calm periods)
- ➡️ Code: `code/shared/12_time_periods.py` 
- **Asset class comparisons** (S&P 500, Gold, Bitcoin, VIX)
- ➡️ Code: `code/shared/13_assets.py` 

---

## 📊 Results

Results are saved in:
- `results/metrics/` → Performance metrics (MAE, RMSE, R², SMAPE, DA)
- `results/plots/` → Forecast visualizations and error analysis

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/JakobTSchneider/encoder-decoder-finance-thesis.git
cd encoder-decoder-finance-thesis
