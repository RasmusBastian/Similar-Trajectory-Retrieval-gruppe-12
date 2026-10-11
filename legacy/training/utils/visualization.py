import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_full_day_forecast(
    actual_full_day: np.ndarray,
    recursive_preds: np.ndarray,
    seq_length: int,
    edges_to_plot: list[int] = [0, 8, 100],
    model_name: str = "LSTM",
    save_path: Optional[str] = None
):
    """
    Plot full-day recursive forecast vs actual for selected edges.
    
    Args:
        actual_full_day: Actual data for full day, shape (288, n_features)
        recursive_preds: Recursive predictions, shape (n_preds, n_features)
        seq_length: Sequence length used for predictions
        edges_to_plot: List of edge indices to plot
        model_name: Name of model for plot title
        save_path: Optional path to save the figure
    """
    # Align predictions with full day (pad with NaN at beginning)
    pred_start_idx = seq_length
    recursive_full_day_aligned = np.full((288, actual_full_day.shape[1]), np.nan)
    recursive_full_day_aligned[pred_start_idx:pred_start_idx + len(recursive_preds)] = recursive_preds

    # Time labels for x-axis
    time_labels_full = pd.date_range("2015-01-07 00:00", periods=288, freq='5min')
    time_labels_display = time_labels_full.strftime("%H:%M")
    tick_indices = np.arange(0, 288, 6)
    tick_labels = [time_labels_display[i] for i in tick_indices]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    edge_names = [f"Edge {idx}" for idx in edges_to_plot]

    for ax, edge_idx, name in zip(axes, edges_to_plot, edge_names):
        actual = actual_full_day[:, edge_idx]
        predicted = recursive_full_day_aligned[:, edge_idx]
        
        # Apply 5-point moving average smoothing to actual values
        actual_smoothed = np.convolve(actual, np.ones(5)/5, mode='same')
        actual_smoothed[:2] = actual[:2]
        actual_smoothed[-2:] = actual[-2:]

        ax.plot(time_labels_full, actual, label="Actual", color="lightblue", linewidth=1.5, alpha=0.5)
        ax.plot(time_labels_full, actual_smoothed, label="Actual (5-point smoothed)", color="blue", linewidth=2.2)
        ax.plot(time_labels_full, predicted, label=f"Recursive Forecast ({model_name})", 
                color="red", linestyle="--", linewidth=2.5, alpha=0.9)

        ax.set_ylabel("Traversal Time (s)", fontsize=12)
        ax.set_title(f"{name} - Edge ID {edge_idx}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)

    # X-axis formatting
    axes[-1].set_xlabel("Time of Day", fontsize=13)
    axes[-1].set_xticks(time_labels_full[tick_indices])
    axes[-1].set_xticklabels(tick_labels, rotation=45)

    # Vertical lines for reference (6 AM, 8 AM, 5 PM, 6 PM)
    for ax in axes:
        ax.axvline(time_labels_full[72], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.axvline(time_labels_full[96], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.axvline(time_labels_full[204], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.axvline(time_labels_full[216], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)

    plt.suptitle(f"Full-Day Recursive Travel Time Forecast vs Actual (Day 7)\n"
                f"{model_name} - 288-Step Ahead Prediction", 
                fontsize=17, fontweight="bold", y=0.96)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_average_traffic_pattern(
    actual_full_day: np.ndarray,
    recursive_preds: np.ndarray,
    training_days_data: list[np.ndarray],
    seq_length: int,
    model_name: str = "LSTM",
    save_path: Optional[str] = None
):
    """
    Plot system-wide average traffic pattern.
    
    Args:
        actual_full_day: Actual data for test day, shape (288, n_features)
        recursive_preds: Recursive predictions, shape (n_preds, n_features)
        training_days_data: List of training day data arrays
        seq_length: Sequence length used for predictions
        model_name: Name of model for plot title
        save_path: Optional path to save the figure
    """
    # Calculate averages
    avg_actual_full_day = np.mean(actual_full_day, axis=1)
    
    pred_start_idx = seq_length
    avg_recursive_preds = np.full(288, np.nan)
    avg_recursive_preds[pred_start_idx:pred_start_idx + len(recursive_preds)] = np.mean(recursive_preds, axis=1)
    
    training_days_avg = np.mean([day.mean(axis=1) for day in training_days_data], axis=0)
    
    # Apply smoothing
    avg_actual_smoothed = np.convolve(avg_actual_full_day, np.ones(5)/5, mode='same')
    avg_actual_smoothed[:2] = avg_actual_full_day[:2]
    avg_actual_smoothed[-2:] = avg_actual_full_day[-2:]
    
    training_avg_smoothed = np.convolve(training_days_avg, np.ones(5)/5, mode='same')
    training_avg_smoothed[:2] = training_days_avg[:2]
    training_avg_smoothed[-2:] = training_days_avg[-2:]
    
    # Time labels
    time_labels_full = pd.date_range("2015-01-07 00:00", periods=288, freq='5min')
    time_labels_display = time_labels_full.strftime("%H:%M")
    tick_indices = np.arange(0, 288, 6)
    tick_labels = [time_labels_display[i] for i in tick_indices]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    ax.plot(time_labels_full, training_days_avg, label="Training Days Avg (Raw)", 
            color="lightgreen", linewidth=1.5, alpha=0.4)
    ax.plot(time_labels_full, training_avg_smoothed, label="Training Days Avg (Smoothed)", 
            color="green", linewidth=2.2, alpha=0.7)
    
    ax.plot(time_labels_full, avg_actual_full_day, label="Day 7 Actual (Raw)", 
            color="lightblue", linewidth=1.5, alpha=0.5)
    ax.plot(time_labels_full, avg_actual_smoothed, label="Day 7 Actual (5-point smoothed)", 
            color="blue", linewidth=2.5)
    
    ax.plot(time_labels_full, avg_recursive_preds, label=f"Recursive Forecast ({model_name})", 
            color="red", linestyle="--", linewidth=2.5, alpha=0.9)
    
    ax.set_ylabel("Average Traversal Time (s)", fontsize=13)
    ax.set_xlabel("Time of Day", fontsize=13)
    ax.set_title("Average Traffic Pattern Across All Edges - Day 7 vs Training Days", 
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.35)
    
    # Add vertical lines for reference
    ax.axvline(time_labels_full[72], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axvline(time_labels_full[96], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axvline(time_labels_full[204], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axvline(time_labels_full[216], color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
    
    ax.set_xticks(time_labels_full[tick_indices])
    ax.set_xticklabels(tick_labels, rotation=45)
    
    # Calculate and display MAPE
    valid_mask = ~np.isnan(avg_recursive_preds)
    if np.any(valid_mask):
        avg_mape = np.mean(np.abs((avg_actual_full_day[valid_mask] - avg_recursive_preds[valid_mask]) / 
                                   np.maximum(np.abs(avg_actual_full_day[valid_mask]), 1e-6))) * 100
        ax.text(0.02, 0.98, f'Average MAPE: {avg_mape:.2f}%', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f"System-Wide Average Traffic Forecast vs Actual\n"
                f"{model_name} Recursive Prediction - 288-Step Ahead", 
                fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_training_metrics(train_losses: list, val_losses: list, val_mapes: list, 
                         best_epoch: int, save_path: Optional[str] = None):
    """
    Plot training and validation metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_mapes: List of validation MAPEs per epoch
        best_epoch: Epoch with best validation loss
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.8, label=f'Best ({best_epoch})')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # MAPE over time
    best_mape = min(val_mapes)
    best_mape_epoch = val_mapes.index(best_mape) + 1
    ax2.plot(val_mapes, color='green', linewidth=2.5)
    ax2.axhline(best_mape, color='red', linestyle='--', alpha=0.8, label=f'Best: {best_mape:.2f}% (epoch {best_mape_epoch})')
    ax2.axvline(best_epoch - 1, color='orange', linestyle=':', alpha=0.6, label=f'Best Loss Epoch: {best_epoch}')
    ax2.set_title('Validation MAPE (%) Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAPE (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics plot to {save_path}")
    
    plt.show()
