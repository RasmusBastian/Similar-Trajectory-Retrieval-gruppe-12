# PCA+LSTM Traffic Prediction Model

This project trains an LSTM model with PCA dimensionality reduction for traffic traversal time prediction.

## Setup

### 1. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run hyperparameter optimization

First, run the hyperparameter optimization script to find the best model configuration:

```bash
python hyperparameter_optimize.py
```

This will run 50 trials by default and save the results to `output/optimization_results/`. When complete, it will print the best hyperparameters found.

### Step 2: Update train.py with best hyperparameters

Open `train.py` and update the hyperparameters in the `if __name__ == "__main__":` block (around line 320) with the values from the optimization:

```python
model, scaler, pca, mse, mape = train_pca_lstm(
    n_components=<best_n_components>,
    seq_length=<best_seq_length>,
    hidden_size=<best_hidden_size>,
    num_layers=<best_num_layers>,
    num_epochs=10000,
    batch_size=<best_batch_size>,
    learning_rate=<best_learning_rate>,
    patience=50,
    dropout=<best_dropout>,
    huber_delta=<best_huber_delta>
)
```

You can find the best parameters in `output/optimization_results/best_params.json`.

### Step 3: Train the final model

```bash
python train.py
```

## Input Data

Place your input data files in the `input/` directory:

- `edge_data_day3.csv` - Training data (day 3)
- `edge_data_day4.csv` - Training data (day 4)
- `edge_data_day5.csv` - Training data (day 5)
- `edge_data_day6.csv` - Training data (day 6)
- `edge_data_day7.csv` - Test data (day 7)

Each CSV should have a `time_slot` column and feature columns for each edge.

## Output Directory

After running the scripts, the `output/` directory will contain:

### From `hyperparameter_optimize.py`:

- `optimization_results/best_params.json` - Best hyperparameters found
- `optimization_results/all_trials.csv` - Results from all optimization trials
- `optimization_results/study_summary.json` - Summary of the optimization study

### From `train.py`:

- `best_model.onnx` - Trained model exported in ONNX format (includes PCA transforms embedded)
- `prediction_plots/pca_lstm_training_metrics.png` - Training and validation loss curves
- `prediction_plots/pca_lstm_full_day.png` - Full-day forecast vs actual for selected edges
- `prediction_plots/pca_lstm_average.png` - System-wide average traffic pattern comparison
