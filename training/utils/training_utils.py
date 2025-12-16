import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple
import platform

from .data_utils import inverse_pca

def get_device() -> torch.device:
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
        print("Using Metal (MPS) acceleration on macOS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA acceleration (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device

def create_data_loaders(
    X_train: np.ndarray, y_train: np.ndarray, time_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, time_test: np.ndarray,
    batch_size: int, device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Args:
        X_train, y_train, time_train: Training data
        X_test, y_test, time_test: Test data
        batch_size: Batch size for data loaders
        device: Device to load tensors onto
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    time_train_t = torch.tensor(time_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    time_test_t = torch.tensor(time_test, dtype=torch.float32).to(device)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, time_train_t),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t, time_test_t),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y, batch_time in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X, batch_time)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def validate_epoch(model: nn.Module, test_loader: DataLoader, 
                   criterion: nn.Module, scaler, pca) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        test_loader: Test data loader
        criterion: Loss function
        scaler: Fitted scaler for inverse PCA
        pca: Fitted PCA for inverse transform
        
    Returns:
        Tuple of (avg_val_loss, mape, predictions, targets) in original space
    """
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_time in test_loader:
            predictions = model(batch_X, batch_time)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
            all_val_preds.append(predictions.cpu().numpy())
            all_val_targets.append(batch_y.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    
    # Calculate MAPE in original space
    val_preds_concat = np.concatenate(all_val_preds, axis=0)
    val_targets_concat = np.concatenate(all_val_targets, axis=0)
    val_preds_orig = inverse_pca(val_preds_concat, scaler, pca)
    val_targets_orig = inverse_pca(val_targets_concat, scaler, pca)
    mape = np.mean(np.abs((val_targets_orig - val_preds_orig) / 
                          np.maximum(np.abs(val_targets_orig), 1e-1))) * 100
    
    return avg_val_loss, mape, val_preds_orig, val_targets_orig


def train_model_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    pca,
    num_epochs: int,
    patience: int
) -> dict:
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        scaler: Fitted scaler for inverse PCA
        pca: Fitted PCA for inverse transform
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history and best model state
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    val_mapes = []
    
    print(f"\nTraining for up to {num_epochs} epochs (patience={patience})...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, mape, _, _ = validate_epoch(model, test_loader, criterion, scaler, pca)
        val_losses.append(val_loss)
        val_mapes.append(mape)
        
        # Print progress
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f} - MAPE: {mape:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mapes': val_mapes,
        'model_state': best_model_state
    }