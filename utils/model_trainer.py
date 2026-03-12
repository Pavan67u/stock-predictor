# ================================================================
# utils/model_trainer.py — Model Training Pipeline
# ================================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore[import-untyped]
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[import-untyped]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore[import-untyped]
from tensorflow.keras.optimizers import Adam  # type: ignore[import-untyped]


def create_sequences(data: np.ndarray, window_size: int = 60):
    """
    Create sliding window sequences for time series prediction.

    Parameters:
        data: Scaled price data (numpy array)
        window_size: Number of past days to use as input

    Returns:
        X: Input sequences of shape (num_samples, window_size)
        y: Target values of shape (num_samples,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build a 3-layer LSTM model for stock price prediction.

    Parameters:
        input_shape: Tuple (time_steps, features) — e.g., (60, 1)

    Returns:
        Compiled Keras Sequential model
    """
    tf.random.set_seed(42)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


def train_lstm(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    """
    Train the LSTM model with early stopping and LR scheduling.

    Parameters:
        model: Compiled Keras model
        X_train: Training input (samples, time_steps, features)
        y_train: Training targets
        epochs: Maximum training epochs
        batch_size: Batch size
        validation_split: Fraction of training data for validation

    Returns:
        Training history object
    """
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6, verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1,
    )

    return history


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a Linear Regression baseline model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train a Random Forest Regression model."""
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_svr(X_train: np.ndarray, y_train: np.ndarray) -> SVR:
    """Train a Support Vector Regression model."""
    model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma="scale")
    model.fit(X_train, y_train)
    return model
