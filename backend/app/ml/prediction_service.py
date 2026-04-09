"""
Prediction Service - Use teacher/student/ONNX models for predictions

Supports 3 inference modes:
1. ONNX (fastest, default) - ~30-50ms
2. Student PyTorch (fast) - ~50-80ms
3. Teacher PyTorch (accurate) - ~100-150ms
"""

import logging
from typing import Dict, Any, Literal
import time
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .model_loader import get_models, get_model_loader

logger = logging.getLogger(__name__)

ModelType = Literal['onnx', 'student', 'teacher']


class PredictionService:
    """High-level prediction service"""
    
    def __init__(self):
        self.models = None
        self.seq_len = 64
        self.n_features = 14
        self._load_models()
    
    def _load_models(self):
        """Load models on initialization"""
        try:
            self.models = get_models()
            logger.info("Prediction service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.models = None
    
    def predict_energy(
        self,
        distance_km: float,
        speed_kmh: float,
        temperature_c: float,
        initial_soc: float,
        initial_soh: float,
        mass_kg: float = 1600,
        drag_coeff: float = 0.28,
        model_type: ModelType = 'onnx'
    ) -> Dict[str, Any]:
        if self.models is None:
            raise RuntimeError("Models not loaded")
        
        start_time = time.time()
        
        # Prepare input
        features = self._prepare_input(
            distance_km, speed_kmh, temperature_c,
            initial_soc, initial_soh, mass_kg, drag_coeff
        )
        
        # Route to appropriate model
        if model_type == 'onnx' and 'onnx_session' in self.models:
            result = self._predict_onnx(features)
        elif model_type == 'student' and 'student' in self.models:
            result = self._predict_student(features)
        elif model_type == 'teacher' and 'teacher' in self.models:
            result = self._predict_teacher(features)
        else:
            # Fallback
            if 'onnx_session' in self.models:
                result = self._predict_onnx(features)
            elif 'student' in self.models:
                result = self._predict_student(features)
            elif 'teacher' in self.models:
                result = self._predict_teacher(features)
            else:
                raise RuntimeError("No models available")
        
        inference_time = (time.time() - start_time) * 1000
        result['inference_time_ms'] = round(inference_time, 2)
        
        return result
    
    def _prepare_input(self, distance_km, speed_kmh, temperature_c,
                       initial_soc, initial_soh, mass_kg, drag_coeff) -> np.ndarray:
        """Prepare and normalize input sequence"""
        
        # Create a simple sequence
        features = np.array([
            400.0,                    # V (voltage)
            10.0,                     # I (current - estimate)
            temperature_c,            # T_batt
            initial_soc / 100.0,      # SoC (normalize)
            initial_soh / 100.0,      # SoH (normalize)
            speed_kmh / 3.6,          # speed_ms
            0.0,                      # elevation (flat)
            0.0,                      # grade (flat)
            10.0,                     # P_total (estimate)
            75.0,                     # capacity_initial (default)
            mass_kg,                  # mass
            drag_coeff,               # Cd
            0.01,                     # Cr (rolling resistance)
            2.5                       # A (frontal area)
        ], dtype=np.float32)
        
        # Replicate to create sequence
        sequence = np.tile(features, (self.seq_len, 1))
        
        # Normalize using scaler
        if 'scaler' in self.models:
            scaler = self.models['scaler']
            sequence_normalized = scaler.transform(sequence)
        else:
            sequence_normalized = sequence
            
        return sequence_normalized.flatten().astype(np.float32)
    
    def _predict_onnx(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict using ONNX (fastest)"""
        session = self.models['onnx_session']
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: features.reshape(1, -1)})
        
        soc_mean, soc_logvar, soh_mean, soh_logvar, energy_mean, energy_logvar = outputs
        
        energy_kwh = float(energy_mean[0])
        final_soc = float(soc_mean[0]) * 100
        final_soh = float(soh_mean[0]) * 100
        
        energy_var = np.exp(float(energy_logvar[0]))
        energy_std = np.sqrt(energy_var)
        confidence = max(0.5, min(0.95, 1.0 - (energy_std / (energy_kwh + 1e-6))))
        
        return {
            'energy_kwh': round(energy_kwh, 2),
            'final_soc': round(final_soc, 1),
            'final_soh': round(final_soh, 1),
            'confidence': round(confidence, 3),
            'model_used': 'onnx'
        }
    
    def _predict_student(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict using Student PyTorch (fast)"""
        if not TORCH_AVAILABLE or self.models.get('student') is None:
            raise RuntimeError("Student model/PyTorch not available")
            
        student = self.models['student']
        device = next(student.parameters()).device
        
        X = torch.from_numpy(features.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            outputs = student(X)
        
        soc_mean, soc_logvar, soh_mean, soh_logvar, energy_mean, energy_logvar = outputs
        
        energy_kwh = energy_mean.cpu().item()
        final_soc = soc_mean.cpu().item() * 100
        final_soh = soh_mean.cpu().item() * 100
        
        energy_var = torch.exp(energy_logvar).cpu().item()
        energy_std = np.sqrt(energy_var)
        confidence = max(0.5, min(0.95, 1.0 - (energy_std / (energy_kwh + 1e-6))))
        
        return {
            'energy_kwh': round(energy_kwh, 2),
            'final_soc': round(final_soc, 1),
            'final_soh': round(final_soh, 1),
            'confidence': round(confidence, 3),
            'model_used': 'student'
        }
    
    def _predict_teacher(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict using Teacher PyTorch (most accurate)"""
        if not TORCH_AVAILABLE or self.models.get('teacher') is None:
            raise RuntimeError("Teacher model/PyTorch not available")
            
        teacher = self.models['teacher']
        device = next(teacher.parameters()).device
        
        X = torch.from_numpy(features.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            outputs = teacher(X)
        
        soc_mean, soc_logvar, soh_mean, soh_logvar, energy_mean, energy_logvar = outputs
        
        energy_kwh = energy_mean.cpu().item()
        final_soc = soc_mean.cpu().item() * 100
        final_soh = soh_mean.cpu().item() * 100
        
        energy_var = torch.exp(energy_logvar).cpu().item()
        energy_std = np.sqrt(energy_var)
        confidence = max(0.5, min(0.95, 1.0 - (energy_std / (energy_kwh + 1e-6))))
        
        return {
            'energy_kwh': round(energy_kwh, 2),
            'final_soc': round(final_soc, 1),
            'final_soh': round(final_soh, 1),
            'confidence': round(confidence, 3),
            'model_used': 'teacher'
        }
    
    def health_check(self) -> Dict[str, Any]:
        try:
            if self.models is None:
                return {'status': 'unhealthy', 'error': 'Models not loaded'}
            
            result = self.predict_energy(
                distance_km=50, speed_kmh=90, temperature_c=25,
                initial_soc=80, initial_soh=95, model_type='onnx'
            )
            
            return {
                'status': 'healthy',
                'models_loaded': list(self.models.keys()),
                'test_prediction': result
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

prediction_service = PredictionService()

def predict_energy(**kwargs) -> Dict[str, Any]:
    return prediction_service.predict_energy(**kwargs)
