# 📈 Stock Market Prediction System

A complete end-to-end machine learning system for stock price prediction using LSTM neural networks, with a FastAPI backend and Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## 🎯 Overview

This project builds a **complete ML pipeline** that:

1. **Collects** real historical stock market data via Yahoo Finance
2. **Engineers** 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. **Trains** 4 ML/DL models (Linear Regression, Random Forest, SVR, LSTM)
4. **Evaluates** models using MSE, RMSE, MAE, and R²
5. **Serves** predictions through a REST API (FastAPI)
6. **Visualizes** results in an interactive dashboard (Streamlit)

## 📊 Models

| Model | Type | Description |
|-------|------|-------------|
| Linear Regression | Baseline | Simple linear relationship |
| Random Forest | Ensemble | 100 trees, max_depth=20 |
| SVR | Kernel-based | RBF kernel, C=100 |
| **LSTM** | Deep Learning | 3-layer (128→64→32), Dropout=0.2 |

## 🏗️ Architecture

```
Data Sources → Preprocessing → Feature Engineering → Model Training
     ↓                                                      ↓
Yahoo Finance                                    Model Evaluation
                                                       ↓
Frontend (Streamlit) ← Prediction API (FastAPI) ← Best Model
```

## 📁 Project Structure

```
stock_prediction_project/
├── data/                    # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/               # Jupyter exploration notebooks
├── models/                  # Saved trained models
│   ├── lstm_model.h5
│   ├── random_forest.pkl
│   ├── svr_model.pkl
│   ├── scaler.pkl
│   └── metrics.json
├── backend/                 # FastAPI REST API
│   ├── app.py
│   ├── models.py
│   ├── predictor.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Streamlit dashboard
│   ├── dashboard.py
│   ├── components/
│   ├── requirements.txt
│   └── Dockerfile
├── utils/                   # Reusable utilities
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   └── evaluator.py
├── tests/                   # Unit tests
├── configs/                 # Configuration files
│   └── config.yaml
├── docker-compose.yml
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── Stock_Market_Prediction_System.ipynb  # Main notebook
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/stock_prediction_project.git
cd stock_prediction_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebook

Open `Stock_Market_Prediction_System.ipynb` in VS Code or Jupyter and run all cells.

### 3. Start the API

```bash
cd backend
uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 4. Launch the Dashboard

```bash
cd frontend
streamlit run dashboard.py
# Dashboard: http://localhost:8501
```

### 5. Docker Deployment

```bash
docker-compose up --build -d
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/stock/{ticker}` | Historical stock data |
| `POST` | `/api/predict` | Next-day price prediction |
| `POST` | `/api/predict/trend` | Multi-day trend forecast |
| `GET` | `/api/metrics` | Model evaluation metrics |
| `GET` | `/api/indicators/{ticker}` | Technical indicators |

## 🛠️ Tech Stack

- **Data:** yfinance, pandas, numpy
- **ML:** scikit-learn (Linear Regression, Random Forest, SVR)
- **Deep Learning:** TensorFlow/Keras (LSTM)
- **Features:** Custom technical indicators
- **Backend:** FastAPI, uvicorn, Pydantic
- **Frontend:** Streamlit, Plotly
- **Deployment:** Docker, docker-compose

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. Stock market predictions are inherently uncertain. This is **not financial advice** and should not be used for actual trading decisions.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
# stock-predictor
