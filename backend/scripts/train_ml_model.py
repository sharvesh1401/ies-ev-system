"""
ML Model Training Script.

Generates training data (if needed), trains the EnergyPredictorNetwork,
and saves the best model with metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch pandas pyarrow numpy")
    DEPS_AVAILABLE = False

from app.ml.data_generator import PhysicsDataGenerator, ScenarioFeatures, ScenarioLabels
from app.ml.models.energy_predictor import (
    EnergyPredictorNetwork, 
    EnergyPredictorTrainer
)


def load_or_generate_data(
    data_dir: str,
    num_scenarios: int,
    seed: int
) -> tuple:
    """Load existing data or generate new training data."""
    data_path = Path(data_dir)
    train_path = data_path / "train.parquet"
    val_path = data_path / "val.parquet"
    test_path = data_path / "test.parquet"
    
    # Check if data exists
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Loading existing training data...")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
    else:
        print("Generating training data...")
        generator = PhysicsDataGenerator(seed=seed)
        generator.generate_and_save_splits(
            n_samples=num_scenarios,
            output_dir=data_dir
        )
        
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
    
    feature_names = ScenarioFeatures.feature_names()
    label_name = "energy_consumption_kwh"  # Primary prediction target
    
    # Extract features and labels
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df[label_name].values.astype(np.float32).reshape(-1, 1)
    
    X_val = val_df[feature_names].values.astype(np.float32)
    y_val = val_df[label_name].values.astype(np.float32).reshape(-1, 1)
    
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_name].values.astype(np.float32).reshape(-1, 1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_loaders(
    train_data: tuple,
    val_data: tuple,
    batch_size: int = 256
) -> tuple:
    """Create PyTorch DataLoaders."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def evaluate_model(
    model: EnergyPredictorNetwork,
    test_data: tuple,
    device: str
) -> dict:
    """Evaluate model on test set."""
    X_test, y_test = test_data
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test).to(device)
        y_tensor = torch.tensor(y_test).to(device)
        
        predictions, log_var = model(X_tensor)
        
        # MSE
        mse = torch.mean((predictions - y_tensor) ** 2).item()
        
        # MAE
        mae = torch.mean(torch.abs(predictions - y_tensor)).item()
        
        # MAPE
        mape = torch.mean(
            torch.abs((predictions - y_tensor) / y_tensor.clamp(min=1e-6))
        ).item() * 100
        
        # Percentage within 10%
        relative_error = torch.abs((predictions - y_tensor) / y_tensor.clamp(min=1e-6))
        within_10_pct = (relative_error <= 0.10).float().mean().item() * 100
        
        # R² score
        ss_res = torch.sum((y_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_tensor - y_tensor.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'within_10_percent': within_10_pct,
        'n_test_samples': len(y_test)
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML energy prediction model")
    
    parser.add_argument("--num-scenarios", type=int, default=100000,
                        help="Number of training scenarios to generate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for training data")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory for model outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    if not DEPS_AVAILABLE:
        sys.exit(1)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load/generate data
    print("\n" + "="*60)
    print("STEP 1: Data Preparation")
    print("="*60)
    
    train_data, val_data, test_data = load_or_generate_data(
        args.data_dir,
        args.num_scenarios,
        args.seed
    )
    
    X_train, y_train = train_data
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(val_data[0])}")
    print(f"Test samples: {len(test_data[0])}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, args.batch_size
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("STEP 2: Model Initialization")
    print("="*60)
    
    model = EnergyPredictorNetwork(
        input_size=17,
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.2,
        predict_uncertainty=True
    )
    
    # Set normalization parameters from training data
    input_mean = X_train.mean(axis=0)
    input_std = X_train.std(axis=0)
    output_mean = y_train.mean(axis=0)
    output_std = y_train.std(axis=0)
    
    model.set_normalization_params(input_mean, input_std, output_mean, output_std)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = EnergyPredictorTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Training
    print("\n" + "="*60)
    print("STEP 3: Training")
    print("="*60)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "energy_predictor.pth"
    
    train_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=str(checkpoint_path)
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("STEP 4: Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    model = EnergyPredictorTrainer.load_checkpoint(str(checkpoint_path), device)
    
    metrics = evaluate_model(model, test_data, device)
    
    print(f"Test MAPE: {metrics['mape']:.2f}%")
    print(f"Test RMSE: {metrics['rmse']:.4f} kWh")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Within 10% of physics: {metrics['within_10_percent']:.1f}%")
    
    # Save metrics
    metrics_path = output_path / "metrics.json"
    metrics_output = {
        'training': {
            'best_epoch': train_result['best_epoch'],
            'best_val_loss': train_result['best_val_loss'],
            'final_val_mape': train_result['final_val_mape']
        },
        'test': metrics,
        'config': {
            'num_scenarios': args.num_scenarios,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'early_stopping_patience': args.early_stopping_patience
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Model saved to: {checkpoint_path}")
    
    # Success criteria check
    print("\n" + "="*60)
    print("VALIDATION CHECK")
    print("="*60)
    
    mape_pass = metrics['mape'] < 5.0
    within_10_pass = metrics['within_10_percent'] >= 95.0
    
    print(f"✓ MAPE < 5%: {'PASS' if mape_pass else 'FAIL'} ({metrics['mape']:.2f}%)")
    print(f"✓ 95%+ within 10%: {'PASS' if within_10_pass else 'FAIL'} ({metrics['within_10_percent']:.1f}%)")
    
    if mape_pass and within_10_pass:
        print("\n✅ All validation criteria PASSED!")
        return 0
    else:
        print("\n⚠️ Some validation criteria not met. Consider:")
        print("  - Generating more training data")
        print("  - Training for more epochs")
        print("  - Adjusting model architecture")
        return 1


if __name__ == "__main__":
    sys.exit(main())
