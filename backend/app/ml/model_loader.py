"""
Model Loader — Load Pre-Trained ML Models.

Loads PyTorch (.pth) and Scikit-learn (.pkl) models from disk.
Models are trained separately in Google Colab by the student.

The system works without model files (physics-only fallback).
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


class ModelLoader:
    """
    Load and manage ML models.

    Models are trained in Google Colab and provided as files:
    - energy_predictor.pth   (PyTorch DNN)
    - driver_classifier.pth  (PyTorch LSTM)
    - traffic_estimator.pkl  (Scikit-learn RF)
    - metrics.json           (Validation results)
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if TORCH_AVAILABLE
            else "cpu"
        )

        # Model instances (loaded on demand)
        self.energy_predictor = None
        self.driver_classifier = None
        self.traffic_estimator = None
        self.metrics: Optional[Dict[str, Any]] = None

        # Status tracking
        self._load_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------

    def load_energy_predictor(self) -> Optional[Any]:
        """
        Load energy prediction DNN from ``energy_predictor.pth``.

        Returns the loaded PyTorch model, or ``None`` if unavailable.
        """
        if not TORCH_AVAILABLE:
            self._load_errors["energy_predictor"] = "PyTorch not installed"
            return None

        model_path = self.models_dir / "energy_predictor.pth"
        if not model_path.exists():
            self._load_errors["energy_predictor"] = f"File not found: {model_path}"
            return None

        try:
            from app.ml.models.energy_predictor import EnergyPredictorNetwork

            model = EnergyPredictorNetwork(
                input_size=17,
                hidden_sizes=[128, 64, 32],
                dropout_rate=0.2,
            )

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.energy_predictor = model
            return model

        except Exception as e:
            self._load_errors["energy_predictor"] = str(e)
            return None

    def load_driver_classifier(self) -> Optional[Any]:
        """
        Load driver behaviour classifier LSTM from ``driver_classifier.pth``.
        """
        if not TORCH_AVAILABLE:
            self._load_errors["driver_classifier"] = "PyTorch not installed"
            return None

        model_path = self.models_dir / "driver_classifier.pth"
        if not model_path.exists():
            self._load_errors["driver_classifier"] = f"File not found: {model_path}"
            return None

        try:
            from app.ml.models.driver_classifier import DriverClassifierLSTM

            model = DriverClassifierLSTM(
                input_dim=2,
                hidden_dim=64,
                num_classes=3,
            )

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.driver_classifier = model
            return model

        except Exception as e:
            self._load_errors["driver_classifier"] = str(e)
            return None

    def load_traffic_estimator(self) -> Optional[Any]:
        """
        Load traffic impact estimator (Random Forest) from ``traffic_estimator.pkl``.
        """
        model_path = self.models_dir / "traffic_estimator.pkl"
        if not model_path.exists():
            self._load_errors["traffic_estimator"] = f"File not found: {model_path}"
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            self.traffic_estimator = model
            return model

        except Exception as e:
            self._load_errors["traffic_estimator"] = str(e)
            return None

    def load_metrics(self) -> Dict[str, Any]:
        """
        Load training metrics from ``metrics.json``.

        Expected keys: test_mape, test_rmse, physics_agreement, calibration_ece
        """
        metrics_path = self.models_dir / "metrics.json"
        if not metrics_path.exists():
            self._load_errors["metrics"] = f"File not found: {metrics_path}"
            return {}

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            self.metrics = metrics
            return metrics
        except Exception as e:
            self._load_errors["metrics"] = str(e)
            return {}

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def load_all(self) -> Dict[str, Any]:
        """
        Attempt to load all models and metrics.

        Returns dict summarising what was loaded.
        """
        self._load_errors.clear()

        results = {
            "energy_predictor": self.load_energy_predictor() is not None,
            "driver_classifier": self.load_driver_classifier() is not None,
            "traffic_estimator": self.load_traffic_estimator() is not None,
            "metrics": bool(self.load_metrics()),
        }

        return {
            "loaded": results,
            "errors": dict(self._load_errors),
            "all_loaded": all(results.values()),
        }

    def verify_models(self) -> Dict[str, Any]:
        """
        Verify loaded models by running a dummy forward pass.

        Returns a dict with verification results per model.
        """
        results: Dict[str, Any] = {}

        # Energy predictor
        if self.energy_predictor is not None and TORCH_AVAILABLE:
            try:
                dummy = torch.randn(1, 17).to(self.device)
                with torch.no_grad():
                    self.energy_predictor(dummy)
                results["energy_predictor"] = {"status": "OK"}
            except Exception as e:
                results["energy_predictor"] = {"status": "FAIL", "error": str(e)}
        else:
            results["energy_predictor"] = {"status": "NOT_LOADED"}

        # Driver classifier
        if self.driver_classifier is not None and TORCH_AVAILABLE:
            try:
                dummy = torch.randn(1, 10, 2).to(self.device)
                with torch.no_grad():
                    self.driver_classifier(dummy)
                results["driver_classifier"] = {"status": "OK"}
            except Exception as e:
                results["driver_classifier"] = {"status": "FAIL", "error": str(e)}
        else:
            results["driver_classifier"] = {"status": "NOT_LOADED"}

        # Traffic estimator
        if self.traffic_estimator is not None:
            try:
                import numpy as np
                dummy = np.array([[8, 1, 0, 50]])
                self.traffic_estimator.predict(dummy)
                results["traffic_estimator"] = {"status": "OK"}
            except Exception as e:
                results["traffic_estimator"] = {"status": "FAIL", "error": str(e)}
        else:
            results["traffic_estimator"] = {"status": "NOT_LOADED"}

        results["all_ok"] = all(
            r.get("status") == "OK" for r in results.values() if isinstance(r, dict) and "status" in r
        )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Return current status of all models."""
        return {
            "models_dir": str(self.models_dir.absolute()),
            "device": str(self.device),
            "energy_predictor_loaded": self.energy_predictor is not None,
            "driver_classifier_loaded": self.driver_classifier is not None,
            "traffic_estimator_loaded": self.traffic_estimator is not None,
            "metrics_loaded": self.metrics is not None,
            "errors": dict(self._load_errors),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_model_loader: Optional[ModelLoader] = None


def get_model_loader(models_dir: str = "models") -> ModelLoader:
    """
    Get / create singleton ModelLoader.

    On first call, attempts to load all models.
    Models are optional — the system works in physics-only mode without them.
    """
    global _model_loader

    if _model_loader is None:
        _model_loader = ModelLoader(models_dir=models_dir)
        _model_loader.load_all()

    return _model_loader
