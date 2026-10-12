import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(csv_path: str):
    """Load traffic data from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame of traffic data, time_slots array
    """
    df = pd.read_csv(csv_path)

    # Extract time_slot column
    time_slots = df['time_slot'].values
    df = df.drop(columns=['time_slot'])


    return df, time_slots


def load_multi_day_data(day_range: list[int], test_day: int, data_path: str = "input"):
    """
    Load multiple days of training data and one test day.
    
    Args:
        day_range: List of day numbers to use for training (e.g., [3, 4, 5, 6])
        test_day: Day number to use for testing (e.g., 7)
        data_path: Directory containing the data files
        
    Returns:
        Tuple of (train_data, test_data, train_time_slots, test_time_slots)
    """
    print(f"Training on days {day_range}, Testing on day {test_day}")
    
    # Load training days
    train_dfs = []
    train_time_slots = []
    for day in day_range:
        day_df, time_slots = load_data(f"{data_path}/edge_data_day{day}.csv")
        train_dfs.append(day_df)
        train_time_slots.append(time_slots)
        print(f"  Training Data for day {day} shape: {day_df.shape}")
    
    print("  " + "-" * 48)
    
    # Load test day
    test_df, test_time_slots = load_data(f"{data_path}/edge_data_day{test_day}.csv")
    print(f"  Test Data for day {test_day} shape: {test_df.shape}")
    
    # Concatenate training days
    train_data = pd.concat(train_dfs, axis=0, ignore_index=True).values
    test_data = test_df.values
    train_time_slots = np.concatenate(train_time_slots)

    
    return train_data, test_data, train_time_slots, test_time_slots


def extract_time_features(time_slots: np.ndarray) -> np.ndarray:
    """
    Extract cyclical time features from time_slot values.

    Creates cyclical encodings for:
    - Hour of day (0-23)
    - Minute of day (0-1439)

    Args:
        time_slots: Array of time slot strings

    Returns:
        Array of shape (n_samples, 4) with cyclical time features:
        [hour_sin, hour_cos, minute_of_day_sin, minute_of_day_cos]
    """
    # Convert to datetime - let pandas infer the format
    timestamps = pd.to_datetime(time_slots, format="mixed")

    # Extract components
    hours = timestamps.hour.values
    minutes_of_day = timestamps.hour.values * 60 + timestamps.minute.values

    # Create cyclical features using sine and cosine
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    minute_sin = np.sin(2 * np.pi * minutes_of_day / 1440)
    minute_cos = np.cos(2 * np.pi * minutes_of_day / 1440)

    # Stack features
    time_features = np.stack([hour_sin, hour_cos, minute_sin, minute_cos], axis=1)

    return time_features


def apply_pca(data: np.ndarray, n_components: int = 50, scaler=None, pca=None, fit=True):
    """
    Apply PCA dimensionality reduction.

    Args:
        data: Input data of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        scaler: Pre-fitted scaler (if fit=False)
        pca: Pre-fitted PCA (if fit=False)
        fit: Whether to fit or just transform

    Returns:
        reduced_data, scaler, pca
    """
    if fit:
        # Fit scaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Fit PCA
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data_scaled)

        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"Reduced from {data.shape[1]} to {n_components} dimensions")
    else:
        # Just transform
        data_scaled = scaler.transform(data)
        reduced_data = pca.transform(data_scaled)

    return reduced_data, scaler, pca


def inverse_pca(reduced_data: np.ndarray, scaler, pca):
    """
    Inverse transform PCA back to original space.

    Args:
        reduced_data: PCA-transformed data
        scaler: Fitted scaler
        pca: Fitted PCA

    Returns:
        original_space_data
    """
    scaled_data = pca.inverse_transform(reduced_data)
    original_data = scaler.inverse_transform(scaled_data)
    return original_data


def create_sequences(data: np.ndarray, seq_length: int, time_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM with time features.

    Args:
        data: Array of shape (n_samples, n_features)
        seq_length: Sequence length
        time_features: Time features for each sample,
                       shape (n_samples, n_time_features)

    Returns:
        X: (n_seq, seq_length, n_features)
        y: (n_seq, n_features)
        time_y: (n_seq, n_time_features) - time features for target timestep
    """
    xs = []
    ys = []
    time_ys = []

    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]           # input sequence
        y = data[i + seq_length]               # target next step
        t = time_features[i + seq_length]      # time of target step

        xs.append(x)
        ys.append(y)
        time_ys.append(t)

    return (
        np.array(xs),
        np.array(ys),
        np.array(time_ys)
    )
