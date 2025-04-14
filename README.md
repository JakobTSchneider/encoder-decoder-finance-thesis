# Encoder-Decoder Models for Financial Time Series Forecasting

This repository contains the complete code, data preparation, and evaluation framework developed for a Bachelor's thesis focusing on the application of encoder-decoder architectures to financial time series forecasting.

---

## 🧱 Model Architecture

The project is structured into several logical blocks, similar to a modular notebook:

### 🔹 Block 1: Imports, Seeds and Configuration
- All relevant libraries (TensorFlow, NumPy, pandas, etc.)
- Reproducibility settings for GPU and CPU
- Helper functions for metrics and volatility segmentation
- Dataset loading (S&P 500 via `yfinance`)
- Sequence creation and data splitting

➡️ Code: `code/model_classic/01_preprocessing.py`

---

### 🔹 Block 2: Classic Encoder-Decoder Model
- LSTM-based sequence-to-sequence model
- Teacher forcing decoder
- Multi-step forecast (t+1 to t+20)
- Early stopping and training callbacks

➡️ Code: `code/model_classic/02_model_classic.py` *(WIP)*

---

### 🔹 Block 3: Hybrid Encoder-Decoder Model
- Bidirectional LSTM encoder
- Dot-product attention mechanism
- Autoregressive decoder structure
- Comparison with classical model

➡️ Code: `code/model_hybrid/03_model_hybrid.py` *(WIP)*

---

## ⚙️ Automated Experiments

A suite of automated tests evaluates both models under varying conditions:

### 🔬 Hyperparameter Search (Hybrid Model)
- Dropout rates, latent dimensions, input/output lengths
- Automated heatmap generation and CSV export

➡️ Code: `code/shared/hyperparameter_search.py` *(WIP)*

### 🧪 Testing Framework
Both models are tested under different experimental setups:

- **Input/Output length variations** (e.g. 20/5, 60/20, etc.)
- **Time period robustness** (e.g. crisis vs. calm periods)
- **Asset class comparisons** (S&P 500, Gold, Bitcoin, VIX)

➡️ Code: `code/shared/test_runner.py` *(WIP)*

---

## 📊 Results

Results are saved in:
- `results/metrics/` → Performance metrics (MAE, RMSE, R², SMAPE, DA)
- `results/plots/` → Forecast visualizations and error analysis

---

## 📘 Academic Context

This repository supports a Bachelor's thesis in economics and data science. The project systematically compares the robustness and generalizability of different encoder-decoder models for forecasting financial time series.

A final PDF of the thesis will be available after submission under `/thesis/`.

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/JakobTSchneider/encoder-decoder-finance-thesis.git
cd encoder-decoder-finance-thesis
