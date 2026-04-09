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

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


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
        self.teacher = None
        self.student = None
        self.scaler = None
        self.driver_classifier = None
        self.traffic_estimator = None
        self.metrics: Optional[Dict[str, Any]] = None
        
        # ONNX Sessions
        self.onnx_energy_predictor = None
        self.onnx_session = None

        # Status tracking
        self._load_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------

    def load_energy_predictor(self) -> Optional[Any]:
        """
        Load energy prediction model.
        
        Prioritizes:
        1. student (1).onnx (if onnxruntime is available)
        2. student.pth
        3. energy_predictor.pth
        """
        # --- Try ONNX first ---
        onnx_path = self.models_dir / "student (1).onnx"
        if ONNX_AVAILABLE and onnx_path.exists():
            try:
                self.onnx_energy_predictor = ort.InferenceSession(
                    str(onnx_path), 
                    providers=['CPUExecutionProvider']
                )
                print(f"  ✓ Loaded ONNX model: {onnx_path.name}")
                return self.onnx_energy_predictor
            except Exception as e:
                self._load_errors["energy_predictor_onnx"] = str(e)

        # --- Fallback to PyTorch ---
        if not TORCH_AVAILABLE:
            self._load_errors["energy_predictor"] = "PyTorch not installed"
            return None

        # Check multiple possible filenames
        paths_to_try = [
            self.models_dir / "student.pth",
            self.models_dir / "energy_predictor.pth",
            self.models_dir / "teacher.pth"
        ]
        
        model_path = None
        for p in paths_to_try:
            if p.exists():
                model_path = p
                break

        if not model_path:
            self._load_errors["energy_predictor"] = f"No .pth model files found in {self.models_dir}"
            return None

        try:
            from app.ml.models.energy_predictor import EnergyPredictorNetwork

            # We try to load using the robust load_checkpoint if it's a full checkpoint,
            # otherwise fall back to state_dict matching.
            try:
                # Try loading as a full checkpoint first
                model = EnergyPredictorNetwork.load_checkpoint(str(model_path), device=str(self.device))
                print(f"  ✓ Loaded checkpoint model: {model_path.name}")
            except Exception:
                # Fallback to default architecture
                model = EnergyPredictorNetwork(
                    input_size=17,
                    hidden_sizes=[128, 64, 32],
                    dropout_rate=0.2,
                )
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                print(f"  ✓ Loaded state_dict model: {model_path.name}")

            self.energy_predictor = model
            return model

        except Exception as e:
            self._load_errors["energy_predictor"] = str(e)
            return None

    def load_scaler(self):
        scaler_path = self.models_dir / "scaler.pkl"
        if scaler_path.exists() and PICKLE_AVAILABLE:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"  ✓ Loaded scaler: {scaler_path.name}")
                return self.scaler
            except Exception as e:
                self._load_errors["scaler"] = str(e)
        return None

    def load_teacher(self):
        if not TORCH_AVAILABLE: return None
        teacher_path = self.models_dir / "teacher.pth"
        if teacher_path.exists():
            try:
                from app.ml.models.tcn_transformer import TeacherModel
                model = TeacherModel(64, 14)
                model.load_state_dict(torch.load(teacher_path, map_location=self.device, weights_only=True))
                model.to(self.device)
                model.eval()
                self.teacher = model
                print(f"  ✓ Loaded teacher: {teacher_path.name}")
                return model
            except Exception as e:
                self._load_errors["teacher"] = str(e)
        return None

    def load_student(self):
        if not TORCH_AVAILABLE: return None
        student_path = self.models_dir / "student.pth"
        if student_path.exists():
            try:
                from app.ml.models.tcn_transformer import StudentModel
                model = StudentModel(64, 14)
                model.load_state_dict(torch.load(student_path, map_location=self.device, weights_only=True))
                model.to(self.device)
                model.eval()
                self.student = model
                print(f"  ✓ Loaded student: {student_path.name}")
                return model
            except Exception as e:
                self._load_errors["student"] = str(e)
        return None

    def load_models(self, load_teacher=True, load_student=True, load_onnx=True) -> Dict[str, Any]:
        """Method strictly required by prediction_service.py"""
        models = {}
        if not self.scaler: self.load_scaler()
        if self.scaler: models['scaler'] = self.scaler
        
        if load_teacher:
            if not self.teacher: self.load_teacher()
            if self.teacher: models['teacher'] = self.teacher
            
        if load_student:
            if not self.student: self.load_student()
            if self.student: models['student'] = self.student
            
        if load_onnx:
            if not self.onnx_session:
                onnx_path = self.models_dir / "student.onnx"
                if ONNX_AVAILABLE and onnx_path.exists():
                    self.onnx_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            if self.onnx_session: models['onnx_session'] = self.onnx_session
            
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        info = {"loaded": True, "device": str(self.device), "models_available": []}
        if self.scaler: info["models_available"].append("scaler")
        if self.teacher: 
            info["models_available"].append("teacher")
            info["teacher_params"] = sum(p.numel() for p in self.teacher.parameters())
        if self.student:
            info["models_available"].append("student")
            info["student_params"] = sum(p.numel() for p in self.student.parameters())
        if self.onnx_session: info["models_available"].append("onnx_session")
        return info

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
            "teacher": self.load_teacher() is not None,
            "student": self.load_student() is not None,
            "scaler": self.load_scaler() is not None,
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
        if self.onnx_energy_predictor is not None:
            results["energy_predictor_onnx"] = {"status": "OK (ONNX)"}
        elif self.energy_predictor is not None and TORCH_AVAILABLE:
            try:
                dummy = torch.randn(1, 17).to(self.device)
                with torch.no_grad():
                    # Check if model has normalize_input, if so use it
                    if hasattr(self.energy_predictor, 'normalize_input'):
                        self.energy_predictor(dummy)
                    else:
                        # Simple forward
                        self.energy_predictor(dummy)
                results["energy_predictor"] = {"status": "OK (PyTorch)"}
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
            "energy_predictor_loaded": self.energy_predictor is not None or self.onnx_energy_predictor is not None,
            "using_onnx": self.onnx_energy_predictor is not None,
            "driver_classifier_loaded": self.driver_classifier is not None,
            "traffic_estimator_loaded": self.traffic_estimator is not None,
            "metrics_loaded": self.metrics is not None,
            "errors": dict(self._load_errors),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_model_loader: Optional[ModelLoader] = None


def get_model_loader(models_dir: str = None) -> ModelLoader:
    """
    Get / create singleton ModelLoader.

    On first call, attempts to load all models.
    Models are optional — the system works in physics-only mode without them.
    """
    global _model_loader

    if _model_loader is None:
        if models_dir is None:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")
        _model_loader = ModelLoader(models_dir=models_dir)
        _model_loader.load_all()

    return _model_loader

def get_models() -> Dict[str, Any]:
    """Helper for prediction_service.py to get models dictionary"""
    loader = get_model_loader()
    return loader.load_models()
