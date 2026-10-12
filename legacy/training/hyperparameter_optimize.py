"""
Optuna hyperparameter optimization for PCA+LSTM traffic prediction model.

Usage:
    python hyperparameter_optimization.py
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

from utils.data_utils import (
    load_multi_day_data, extract_time_features, apply_pca, create_sequences
)
from utils.models import LSTMForecasterWithDropout
from utils.training_utils import get_device, create_data_loaders, train_epoch, validate_epoch


class EarlyStoppingCallback:
    """Optuna callback for early stopping based on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_value = trial.value
        if current_value is None:
            return

        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            study.stop()


def objective(trial: Trial, device: torch.device, train_data: np.ndarray,
              test_data: np.ndarray, train_time_slots: np.ndarray,
              test_time_slots: np.ndarray) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        device: Torch device
        train_data: Training data
        test_data: Test data
        train_time_slots: Training time slots
        test_time_slots: Test time slots

    Returns:
        Validation MAPE (to minimize)
    """
    # Hyperparameters to optimize (aggressive wide ranges)
    n_components = trial.suggest_int("n_components", 50, 300, step=10)
    seq_length = trial.suggest_int("seq_length", 3, 30, step=3)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 768, 1024, 1536, 2048])
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.7, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0, step=0.1)

    # Extract time features
    train_time_features = extract_time_features(train_time_slots)
    test_time_features = extract_time_features(test_time_slots)

    # Apply PCA
    train_reduced, scaler, pca = apply_pca(train_data, n_components=n_components, fit=True)
    test_reduced, _, _ = apply_pca(test_data, n_components=n_components,
                                    scaler=scaler, pca=pca, fit=False)

    # Create sequences
    X_train, y_train, time_train = create_sequences(train_reduced, seq_length, train_time_features)
    X_test, y_test, time_test = create_sequences(test_reduced, seq_length, test_time_features)

    # Check if we have enough data
    if len(X_train) < batch_size or len(X_test) < batch_size:
        return float('inf')

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, time_train,
        X_test, y_test, time_test,
        batch_size, device
    )

    # Create model
    input_size = n_components
    output_size = n_components
    time_feature_size = time_train.shape[1]

    model = LSTMForecasterWithDropout(
        input_size, hidden_size, num_layers, output_size,
        dropout=dropout, time_feature_size=time_feature_size
    )
    model.to(device)

    # Training setup
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 15
    epochs_no_improve = 0
    max_epochs = 100

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, mape, _, _ = validate_epoch(model, test_loader, criterion, scaler, pca)

        # Report intermediate value for pruning
        trial.report(mape, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mape = mape
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_mape


def run_optimization(n_trials: int = 100, study_name: str = "pca_lstm_optimization",
                     storage: str = None) -> optuna.Study:
    """
    Run hyperparameter optimization.

    Args:
        n_trials: Number of trials to run
        study_name: Name for the study
        storage: Optional database URL for persistent storage

    Returns:
        Completed Optuna study
    """
    device = get_device()

    # Load data once (outside objective function)
    print("\nLoading data for optimization...")
    train_data, test_data, train_time_slots, test_time_slots = load_multi_day_data(
        day_range=[3, 4, 5, 6],
        test_day=7,
        data_path="input"
    )
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Create objective with data
    def objective_with_data(trial: Trial) -> float:
        return objective(trial, device, train_data, test_data,
                        train_time_slots, test_time_slots)

    # Create or load study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    # Run optimization
    print(f"\nStarting optimization with {n_trials} trials...")
    print("=" * 70)

    study.optimize(
        objective_with_data,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[EarlyStoppingCallback(patience=20)]
    )

    return study


def print_results(study: optuna.Study) -> None:
    """Print optimization results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best MAPE: {study.best_value:.2f}%")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Show top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number'])}: MAPE={row['value']:.2f}%")


def save_results(study: optuna.Study, output_dir: str = "output/optimization_results") -> None:
    """Save optimization results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save best params as JSON
    best_params_path = output_path / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_params_path}")

    # Save all trials as CSV
    trials_df = study.trials_dataframe()
    trials_path = output_path / "all_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved to {trials_path}")

    # Save study summary
    summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    summary_path = output_path / "study_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Study summary saved to {summary_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("PCA+LSTM HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # Run optimization
    study = run_optimization(
        n_trials=50,
        study_name="pca_lstm_study"
    )

    # Print and save results
    print_results(study)
    save_results(study)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
