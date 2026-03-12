# ================================================================
# utils/evaluator.py — Model Evaluation & Visualization
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Parameters:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary with MSE, RMSE, MAE, and R² scores
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def compare_models(model_results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from model results.

    Parameters:
        model_results: Dict mapping model names to their metrics dicts

    Returns:
        Sorted DataFrame comparing all models
    """
    rows = []
    for name, metrics in model_results.items():
        rows.append({
            "Model": name,
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R²": metrics["R2"],
        })

    df = pd.DataFrame(rows).sort_values("RMSE")
    return df


def inverse_scale(predictions: np.ndarray, scaler) -> np.ndarray:
    """Convert scaled predictions back to original price range."""
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


def plot_predictions(
    y_test_dollars: np.ndarray,
    model_results: dict,
    scaler,
    ticker: str = "AAPL",
):
    """
    Plot predicted vs actual prices for all models.

    Parameters:
        y_test_dollars: Actual test prices in dollar values
        model_results: Dict mapping model names to dicts with 'predictions' key
        scaler: Fitted MinMaxScaler for inverse transformation
        ticker: Stock ticker for chart title
    """
    model_list = list(model_results.keys())
    colors = ["#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (model_name, color) in enumerate(zip(model_list, colors)):
        ax = axes[idx // 2][idx % 2]
        preds = model_results[model_name]["predictions"]
        preds_dollars = inverse_scale(preds, scaler)

        ax.plot(y_test_dollars, label="Actual Price", color="#2196F3", linewidth=1.5, alpha=0.8)
        ax.plot(preds_dollars, label=f"{model_name}", color=color, linewidth=1.2, alpha=0.9)
        ax.set_title(f"{model_name} — Predicted vs Actual", fontsize=13, fontweight="bold")
        ax.set_xlabel("Test Day Index")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        r2 = model_results[model_name]["R2"]
        ax.annotate(
            f"R² = {r2:.4f}", xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=12, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8),
        )

    plt.suptitle(f"{ticker} — Model Predictions vs Actual Test Data", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
