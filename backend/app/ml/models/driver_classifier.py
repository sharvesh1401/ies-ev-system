"""
Driver Classifier LSTM Neural Network.

Classifies driver behavior into aggressive, moderate, or eco
from speed and acceleration time-series data.

Architecture matches what is trained in Google Colab.
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


if TORCH_AVAILABLE:
    class DriverClassifierLSTM(nn.Module):
        """
        LSTM-based driver behavior classifier.

        Input: Time-series of (speed, acceleration) pairs.
               Shape: (batch_size, sequence_length, 2)
        Output: Class probabilities for 3 driver styles.
               Shape: (batch_size, 3)

        Classes:
            0 = aggressive
            1 = moderate
            2 = eco
        """

        STYLE_NAMES = ["aggressive", "moderate", "eco"]
        INPUT_DIM = 2      # speed, acceleration
        NUM_CLASSES = 3     # aggressive, moderate, eco

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden_dim: int = 64,
            num_layers: int = 2,
            num_classes: int = NUM_CLASSES,
            dropout_rate: float = 0.3,
            bidirectional: bool = False,
        ):
            """
            Initialize driver classifier.

            Args:
                input_dim: Number of input features per timestep (2: speed, accel)
                hidden_dim: LSTM hidden state size
                num_layers: Number of LSTM layers
                num_classes: Number of output classes
                dropout_rate: Dropout probability
                bidirectional: Whether to use bidirectional LSTM
            """
            super().__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            # Classification head
            fc_input_dim = hidden_dim * self.num_directions
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(fc_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch_size, seq_len, input_dim)

            Returns:
                Class logits of shape (batch_size, num_classes)
            """
            # LSTM
            lstm_out, (h_n, _) = self.lstm(x)

            # Use final hidden state from last layer
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                h_forward = h_n[-2]  # Last layer, forward
                h_backward = h_n[-1]  # Last layer, backward
                hidden = torch.cat([h_forward, h_backward], dim=1)
            else:
                hidden = h_n[-1]  # Last layer hidden state

            # Classify
            logits = self.classifier(hidden)
            return logits

        def predict_style(self, x: torch.Tensor) -> dict:
            """
            Predict driver style with probabilities.

            Args:
                x: Input tensor of shape (batch_size, seq_len, 2)

            Returns:
                Dict with 'style' (str), 'style_index' (int),
                and 'probabilities' (dict mapping style name to probability)
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)

                style_idx = torch.argmax(probs, dim=-1).item()
                style_name = self.STYLE_NAMES[style_idx]

                prob_dict = {
                    name: probs[0, i].item()
                    for i, name in enumerate(self.STYLE_NAMES)
                }

            return {
                "style": style_name,
                "style_index": style_idx,
                "probabilities": prob_dict,
            }

else:
    # Stub when PyTorch is not available
    class DriverClassifierLSTM:
        """Stub: PyTorch not available."""

        STYLE_NAMES = ["aggressive", "moderate", "eco"]

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for DriverClassifierLSTM. "
                "Install with: pip install torch"
            )
