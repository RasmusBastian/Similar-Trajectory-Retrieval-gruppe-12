import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from utils.data_utils import (
    load_multi_day_data, extract_time_features, apply_pca,
    inverse_pca, create_sequences, load_data
)
from utils.models import LSTMForecasterWithDropout
from utils.training_utils import (
    get_device, create_data_loaders, train_model_with_early_stopping,
    validate_epoch
)
from utils.visualization import (
    plot_full_day_forecast, plot_average_traffic_pattern,
    plot_training_metrics
)

def perform_recursive_forecasting(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_time_slots: np.ndarray,
    test_data_full: np.ndarray,
    seq_length: int,
    device: torch.device,
    scaler,
    pca,
    verbose: bool = True
) -> np.ndarray:
    """
    Perform recursive multi-step forecasting.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test targets
        test_time_slots: Time slot strings for test period
        test_data_full: Full test day data (original space, all 288 timesteps)
        seq_length: Sequence length
        device: Device to run on
        scaler: Fitted scaler
        pca: Fitted PCA
        verbose: Whether to print debug info
        
    Returns:
        Predictions in original space, shape (n_predictions, n_features)
    """
    model.eval()
    
    initial_input = torch.tensor(X_test[0:1], dtype=torch.float32).to(device)
    initial_timestamp = pd.to_datetime(test_time_slots[seq_length])
    
    recursive_preds_pca = []
    current_input = initial_input
    
    if verbose:
        print("Performing recursive forecasting for entire test period...")
        print(f"Starting recursive forecast with seq_length={seq_length}")
        print(f"Initial timestamp: {initial_timestamp}")
        print(f"Total steps to predict: {len(y_test)}\n")
    
    for step in range(len(y_test)):
        with torch.no_grad():
            # Generate time features for current prediction step
            future_timestamp = initial_timestamp + pd.Timedelta(minutes=5*step)
            current_time_features = extract_time_features(
                np.array([future_timestamp.strftime('%Y-%m-%d %H:%M:%S')])
            )
            current_time = torch.tensor(current_time_features, dtype=torch.float32).to(device)
            
            next_pred = model(current_input, current_time)
            pred_pca = next_pred.cpu().numpy()[0]
            recursive_preds_pca.append(pred_pca)
            
            # Debug output for selected steps
            if verbose and (step < 40 or step == len(y_test) - 1):
                pred_orig = inverse_pca(pred_pca.reshape(1, -1), scaler, pca)[0]
                
                # Get actual value from the full test data at the correct position
                # The prediction at step 0 corresponds to position seq_length in the full data
                actual_position = seq_length + step
                actual_orig = test_data_full[actual_position]
                
                num_actual_in_seq = max(0, seq_length - step)
                num_pred_in_seq = min(step, seq_length)
                
                print(f"\n{'='*80}")
                print(f"Step {step+1}/{len(y_test)} | Timestamp: {future_timestamp} | Position in full day: {actual_position}")
                print(f"  Input sequence composition: {num_actual_in_seq} actual + {num_pred_in_seq} predicted values")
                print(f"  Predicting for edge 0:")
                print(f"    Predicted: {pred_orig[0]:.2f}s")
                print(f"    Actual:    {actual_orig[0]:.2f}s (from position {actual_position} in full day data)")
                print(f"    Error:     {abs(pred_orig[0] - actual_orig[0]):.2f}s "
                      f"({abs(pred_orig[0] - actual_orig[0])/max(actual_orig[0], 1e-6)*100:.1f}%)")
            
            # Update input sequence
            new_sequence = current_input[:, 1:, :]
            new_timestep = next_pred.unsqueeze(1)
            current_input = torch.cat((new_sequence, new_timestep), dim=1)
    
    # Transform back to original space
    recursive_preds_pca = np.array(recursive_preds_pca)
    recursive_preds_original = inverse_pca(recursive_preds_pca, scaler, pca)
    
    return recursive_preds_original


class PCALSTMWrapper(nn.Module):
    """Wrapper that includes PCA transform and inverse transform with the LSTM model."""

    def __init__(self, lstm_model: nn.Module, scaler_mean: np.ndarray, scaler_scale: np.ndarray,
                 pca_components: np.ndarray):
        super().__init__()
        self.lstm = lstm_model

        # Register PCA/scaler parameters as buffers (not trainable)
        self.register_buffer('scaler_mean', torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer('scaler_scale', torch.tensor(scaler_scale, dtype=torch.float32))
        self.register_buffer('pca_components', torch.tensor(pca_components, dtype=torch.float32))

    def pca_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from original space (21312) to PCA space (100)."""
        # x shape: (batch, seq_len, n_features) or (batch, n_features)
        scaled = (x - self.scaler_mean) / self.scaler_scale
        return scaled @ self.pca_components.T

    def pca_inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from PCA space (100) back to original space (21312)."""
        scaled = x @ self.pca_components
        return scaled * self.scaler_scale + self.scaler_mean

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: PCA transform -> LSTM -> PCA inverse transform.

        Args:
            x: Input sequence in original space, shape (batch, seq_len, 21312)
            time_features: Time features, shape (batch, 4)

        Returns:
            Prediction in original space, shape (batch, 21312)
        """
        # Transform input sequence to PCA space
        x_pca = self.pca_transform(x)

        # Run LSTM in PCA space
        pred_pca = self.lstm(x_pca, time_features)

        # Transform prediction back to original space
        pred_original = self.pca_inverse_transform(pred_pca)

        return pred_original


def train_pca_lstm(
    n_components: int = 50,
    seq_length: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    patience: int = 15,
    dropout: float = 0.2,
    huber_delta: float = 1.0
):
    """
    Train PCA+LSTM model for traffic prediction.

    Args:
        n_components: Number of PCA components
        seq_length: Sequence length for LSTM
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        num_epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patience: Early stopping patience
        dropout: Dropout rate

    Returns:
        Tuple of (model, scaler, pca, mse, mape)
    """
    device = get_device()
    
    # Load data
    print("\nLoading multi-day data...")
    train_data, test_data, train_time_slots, test_time_slots = load_multi_day_data(
        day_range=[3, 4, 5, 6],
        test_day=7,
        data_path="input"
    )
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Extract time features
    print("\nCalculating time features...")
    train_time_features = extract_time_features(train_time_slots)
    test_time_features = extract_time_features(test_time_slots)
    print(f"Time features shape: {train_time_features.shape} [hour_sin, hour_cos, minute_sin, minute_cos]")
    print(f"Sample time features (first 5):\n{train_time_features[:5]}")
    
    # Apply PCA
    print(f"\nApplying PCA (reducing to {n_components} components)...")
    train_reduced, scaler, pca = apply_pca(train_data, n_components=n_components, fit=True)
    test_reduced, _, _ = apply_pca(test_data, n_components=n_components, scaler=scaler, pca=pca, fit=False)
    
    # Create sequences
    print(f"\nCreating sequences (seq_length={seq_length})...")
    X_train, y_train, time_train = create_sequences(train_reduced, seq_length, train_time_features)
    X_test, y_test, time_test = create_sequences(test_reduced, seq_length, test_time_features)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, time_train,
        X_test, y_test, time_test,
        batch_size, device
    )
    print(f"\nData loaders created with batch size {batch_size}.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n" + "="*70)
    print("MODEL TRAINING")

    # Create model
    print(f"Creating LSTM model (hidden_size={hidden_size}, layers={num_layers}, dropout={dropout})...")
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
    print(f"Max epochs: {num_epochs}, Learning rate: {learning_rate}, Patience: {patience}, Optimizer: Adam, Loss: HuberLoss(delta={huber_delta})")
    
    # Train model
    training_history = train_model_with_early_stopping(
        model, train_loader, test_loader, optimizer, criterion,
        scaler, pca, num_epochs, patience
    )
    print("="*70)
    
    # Evaluate
    print("MODEL EVALUATION")
    _, mape, preds_original, targets_original = validate_epoch(
        model, test_loader, criterion, scaler, pca
    )
    mse_original = mean_squared_error(targets_original, preds_original)
    
    print(f"\nTraining Results:")
    print(f"  MSE:  {mse_original:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print("="*70)
    
    # Perform recursive forecasting
    print("LOOKING AT DATA AND VISUALIZATIONS")
    
    # Load all days for visualization (need test day data for debug output)
    print("\nLoading all days for visualization...")
    all_days_data = []
    for day in [3, 4, 5, 6, 7]:
        day_df, _ = load_data(f"input/edge_data_day{day}.csv")
        all_days_data.append(day_df.values)
    
    recursive_preds = perform_recursive_forecasting(
        model, X_test, y_test, test_time_slots, all_days_data[4], seq_length,
        device, scaler, pca, verbose=True
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_metrics(
        training_history['train_losses'],
        training_history['val_losses'],
        training_history['val_mapes'],
        training_history['best_epoch'],
        save_path="output/prediction_plots/pca_lstm_training_metrics.png"
    )

    plot_full_day_forecast(
        actual_full_day=all_days_data[4],
        recursive_preds=recursive_preds,
        seq_length=seq_length,
        edges_to_plot=[0, 8, 100],
        model_name="PCA+LSTM",
        save_path="output/prediction_plots/pca_lstm_full_day.png"
    )

    plot_average_traffic_pattern(
        actual_full_day=all_days_data[4],
        recursive_preds=recursive_preds,
        training_days_data=all_days_data[:4],
        seq_length=seq_length,
        model_name="PCA+LSTM",
        save_path="output/prediction_plots/pca_lstm_average.png"
    )


    return model, scaler, pca, mse_original, mape


if __name__ == "__main__":
    print("=" * 70)
    print("PCA+LSTM MODEL")
    print("=" * 70)

    N_COMPONENTS = 50
    SEQ_LENGTH = 21

    model, scaler, pca, mse, mape = train_pca_lstm(
        n_components=N_COMPONENTS,
        seq_length=SEQ_LENGTH,
        hidden_size=256,
        num_layers=2,
        num_epochs=10000,
        batch_size=8,
        learning_rate=0.03717371747825049,
        patience=50,
        dropout=0.1,
        huber_delta=1.1
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Final MSE:      {mse:.6f}")
    print(f"  Final MAPE:     {mape:.2f}%")
    print("=" * 70)

    # Export to ONNX with PCA embedded
    print("\nExporting model to ONNX (with PCA transforms embedded)...")

    # Create wrapper that includes PCA transforms
    wrapper = PCALSTMWrapper(
        lstm_model=model,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_components=pca.components_
    )
    wrapper.eval()

    # Move wrapper to same device as model
    device = next(model.parameters()).device
    wrapper.to(device)

    # Create dummy inputs in ORIGINAL space (not PCA space)
    # x: (batch_size, seq_length, n_features) - sequence input in original space
    # time_features: (batch_size, 4) - time features
    batch_size = 1
    time_feature_size = 4
    n_original_features = len(scaler.mean_)  # Get actual feature count from scaler

    dummy_x = torch.randn(batch_size, SEQ_LENGTH, n_original_features, device=device)
    dummy_time_features = torch.randn(batch_size, time_feature_size, device=device)

    onnx_path = "output/best_model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_time_features),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['sequence', 'time_features'],
        output_names=['prediction'],
        dynamo=False
    )
    print(f"Model exported to {onnx_path}")
    print(f"  Input 'sequence': shape (1, {SEQ_LENGTH}, {n_original_features})")
    print(f"  Input 'time_features': shape (1, {time_feature_size})")
    print(f"  Output 'prediction': shape (1, {n_original_features})")
    print("=" * 70)
