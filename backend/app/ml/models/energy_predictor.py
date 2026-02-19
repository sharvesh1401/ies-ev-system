"""
Energy Predictor Neural Network.

PyTorch neural network for predicting EV energy consumption
with uncertainty quantification via Monte Carlo Dropout.
"""

from typing import Tuple, Optional, Dict, Any
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

import numpy as np


class EnergyPredictorNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural network for EV energy prediction with uncertainty.
    
    Architecture:
        Input (17) → Linear(128) → ReLU → BatchNorm → Dropout(0.2)
                  → Linear(64)  → ReLU → BatchNorm → Dropout(0.2)
                  → Linear(32)  → ReLU → BatchNorm → Dropout(0.2)
                  → Linear(2)   # [mean, log_variance]
    
    Uses Monte Carlo Dropout for uncertainty estimation.
    """
    
    INPUT_SIZE = 17  # 17 input features
    HIDDEN_SIZES = [128, 64, 32]
    OUTPUT_SIZE = 4  # energy, duration, final_soc, avg_speed (we can choose to predict just energy)
    
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: list = None,
        output_size: int = 1,  # Predict energy with uncertainty (mean + log_var)
        dropout_rate: float = 0.2,
        predict_uncertainty: bool = True
    ):
        """
        Initialize the energy predictor network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output values (usually 1 for energy)
            dropout_rate: Dropout probability
            predict_uncertainty: If True, output (mean, log_var), else just mean
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for EnergyPredictorNetwork")
            
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or self.HIDDEN_SIZES
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.predict_uncertainty = predict_uncertainty
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        if predict_uncertainty:
            # Mean and log-variance outputs
            self.mean_head = nn.Linear(prev_size, output_size)
            self.logvar_head = nn.Linear(prev_size, output_size)
        else:
            self.output_head = nn.Linear(prev_size, output_size)
        
        # Input normalization parameters (to be set from training data)
        self.register_buffer('input_mean', torch.zeros(input_size))
        self.register_buffer('input_std', torch.ones(input_size))
        self.register_buffer('output_mean', torch.zeros(output_size))
        self.register_buffer('output_std', torch.ones(output_size))
        
    def set_normalization_params(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray
    ):
        """Set normalization parameters from training data."""
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)
        self.input_std = torch.clamp(self.input_std, min=1e-6)  # Avoid division by zero
        
        self.output_mean = torch.tensor(output_mean, dtype=torch.float32)
        self.output_std = torch.tensor(output_std, dtype=torch.float32)
        self.output_std = torch.clamp(self.output_std, min=1e-6)
        
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features."""
        return (x - self.input_mean) / self.input_std
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output predictions."""
        return y * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            If predict_uncertainty:
                (mean, log_variance) both of shape (batch_size, output_size)
            Else:
                (prediction, None)
        """
        # Normalize input
        x_norm = self.normalize_input(x)
        
        # Feature extraction
        features = self.feature_extractor(x_norm)
        
        if self.predict_uncertainty:
            mean_norm = self.mean_head(features)
            log_var = self.logvar_head(features)
            
            # Denormalize mean prediction
            mean = self.denormalize_output(mean_norm)
            
            return mean, log_var
        else:
            output_norm = self.output_head(features)
            output = self.denormalize_output(output_norm)
            return output, None
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation.
        
        Runs multiple forward passes with dropout enabled
        to estimate epistemic (model) uncertainty.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with:
                - 'mean': Mean prediction
                - 'std': Total uncertainty (epistemic + aleatoric)
                - 'epistemic_std': Model uncertainty
                - 'aleatoric_std': Data uncertainty (from log_var)
        """
        # Ensure dropout is active
        self.train()
        
        means = []
        log_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_var = self.forward(x)
                means.append(mean)
                if log_var is not None:
                    log_vars.append(log_var)
        
        self.eval()
        
        # Stack samples
        means = torch.stack(means, dim=0)  # (n_samples, batch_size, output_size)
        
        # Epistemic uncertainty: variance of means across samples
        mean_prediction = means.mean(dim=0)
        epistemic_var = means.var(dim=0)
        
        results = {
            'mean': mean_prediction,
            'epistemic_std': torch.sqrt(epistemic_var)
        }
        
        # Aleatoric uncertainty: average of predicted variances
        if log_vars:
            log_vars = torch.stack(log_vars, dim=0)
            aleatoric_var = torch.exp(log_vars).mean(dim=0)
            results['aleatoric_std'] = torch.sqrt(aleatoric_var)
            
            # Total uncertainty
            total_var = epistemic_var + aleatoric_var
            results['std'] = torch.sqrt(total_var)
        else:
            results['aleatoric_std'] = torch.zeros_like(mean_prediction)
            results['std'] = results['epistemic_std']
        
        return results
    
    def get_confidence(
        self,
        x: torch.Tensor,
        n_samples: int = 30
    ) -> torch.Tensor:
        """
        Calculate confidence score from uncertainty.
        
        Higher confidence = lower relative uncertainty.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Confidence scores in [0, 1] range
        """
        results = self.predict_with_uncertainty(x, n_samples)
        
        mean = results['mean']
        std = results['std']
        
        # Coefficient of variation (relative uncertainty)
        # Avoid division by zero
        mean_abs = torch.abs(mean).clamp(min=1e-6)
        cv = std / mean_abs
        
        # Convert to confidence: high CV = low confidence
        # Using sigmoid-like transformation
        confidence = torch.exp(-cv)
        
        return confidence


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood Loss.
    
    This loss learns both the mean and variance of predictions,
    allowing the model to express uncertainty.
    
    Args:
        mean: Predicted mean (batch_size, output_size)
        log_var: Predicted log variance (batch_size, output_size)
        target: Ground truth (batch_size, output_size)
        
    Returns:
        Scalar loss value
    """
    # Clamp log_var for numerical stability
    log_var = torch.clamp(log_var, min=-10, max=10)
    
    # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
    #     = 0.5 * (log_var + (y - mu)^2 * exp(-log_var))
    precision = torch.exp(-log_var)
    squared_error = (target - mean) ** 2
    
    loss = 0.5 * (log_var + precision * squared_error)
    
    return loss.mean()


class EnergyPredictorTrainer:
    """
    Trainer for EnergyPredictorNetwork.
    
    Handles training loop, validation, early stopping,
    and model checkpointing.
    """
    
    def __init__(
        self,
        model: 'EnergyPredictorNetwork',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The network to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: 'cuda', 'cpu', or None for auto-detect
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mape': []
        }
        
    def _compute_loss(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        mean, log_var = self.model(batch_x)
        
        if log_var is not None:
            loss = gaussian_nll_loss(mean, log_var, batch_y)
        else:
            loss = F.mse_loss(mean, batch_y)
            
        return loss
    
    def _compute_mape(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Mean Absolute Percentage Error."""
        # Avoid division by zero
        targets_safe = torch.abs(targets).clamp(min=1e-6)
        mape = torch.abs((predictions - targets) / targets_safe).mean()
        return mape.item() * 100
    
    def train_epoch(
        self,
        train_loader: 'torch.utils.data.DataLoader'
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch_x, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(
        self,
        val_loader: 'torch.utils.data.DataLoader'
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                loss = self._compute_loss(batch_x, batch_y)
                total_loss += loss.item()
                
                mean, _ = self.model(batch_x)
                all_preds.append(mean)
                all_targets.append(batch_y)
        
        avg_loss = total_loss / len(val_loader)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mape = self._compute_mape(all_preds, all_targets)
        
        return avg_loss, mape
    
    def fit(
        self,
        train_loader: 'torch.utils.data.DataLoader',
        val_loader: 'torch.utils.data.DataLoader',
        epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_path: str = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            early_stopping_patience: Stop if no improvement
            checkpoint_path: Path to save best model
            
        Returns:
            Training history and final metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_mape = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_mape'].append(val_mape)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val MAPE={val_mape:.2f}%")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                    print(f"  → Saved best model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_val_mape': self.history['val_mape'][-1],
            'history': self.history
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_mean': self.model.input_mean,
            'input_std': self.model.input_std,
            'output_mean': self.model.output_mean,
            'output_std': self.model.output_std,
            'config': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'output_size': self.model.output_size,
                'dropout_rate': self.model.dropout_rate,
                'predict_uncertainty': self.model.predict_uncertainty
            }
        }, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = None) -> 'EnergyPredictorNetwork':
        """Load model from checkpoint."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
            
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint['config']
        model = EnergyPredictorNetwork(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size'],
            dropout_rate=config['dropout_rate'],
            predict_uncertainty=config['predict_uncertainty']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.input_mean = checkpoint['input_mean']
        model.input_std = checkpoint['input_std']
        model.output_mean = checkpoint['output_mean']
        model.output_std = checkpoint['output_std']
        
        model.to(device)
        model.eval()
        
        return model
