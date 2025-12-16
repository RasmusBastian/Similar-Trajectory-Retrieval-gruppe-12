import torch
import torch.nn as nn


class LSTMForecasterWithDropout(nn.Module):
    """LSTM model with dropout for regularization and optional time features."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int,
                 dropout: float = 0.2, time_feature_size: int = 0) -> None:
        super(LSTMForecasterWithDropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_feature_size = time_feature_size

        # LSTM with dropout between layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Additional dropout layer after LSTM
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer - input is LSTM output + time features (if provided)
        fc_input_size = hidden_size + time_feature_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Traffic sequence data (batch, seq_len, input_size)
            time_features: Time features for target timestep (batch, time_feature_size)
        """
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_time_step_out)

        # Concatenate time features
        combined = torch.cat([dropped, time_features], dim=1)

        # Final prediction
        prediction = self.fc(combined)

        return prediction
