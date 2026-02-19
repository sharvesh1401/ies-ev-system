"""
Hybrid ML-Physics Prediction System.

Routes predictions between fast ML inference and accurate physics simulation
based on scenario complexity and confidence levels.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import math

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    TORCH_AVAILABLE = False

from app.simulation.schemas import (
    VehicleParameters,
    RouteParameters,
    EnvironmentParameters,
    SimulationResult
)
from app.simulation.integrated_engine import IntegratedSimulationEngine


class PredictionMethod(Enum):
    """Method used for prediction."""
    ML_ONLY = "ml"
    PHYSICS_ONLY = "physics"
    ML_VALIDATED = "ml_validated"
    PHYSICS_FALLBACK = "physics_fallback"


@dataclass
class ScenarioInput:
    """Input scenario for prediction."""
    vehicle: VehicleParameters
    route: RouteParameters
    environment: EnvironmentParameters
    initial_soc: float = 90.0
    driver_aggression: float = 0.5
    
    def to_feature_array(self) -> 'np.ndarray':
        """Convert to 17-feature array for ML model."""
        if np is None:
            raise ImportError("NumPy required for ML features")
        
        # Calculate elevation metrics from profile
        elev_gain = 0.0
        elev_loss = 0.0
        if self.route.elevation_profile:
            prev_elev = self.route.elevation_profile[0][1] if self.route.elevation_profile else 0
            for _, elev in self.route.elevation_profile[1:]:
                delta = elev - prev_elev
                if delta > 0:
                    elev_gain += delta
                else:
                    elev_loss += abs(delta)
                prev_elev = elev
        
        return np.array([
            self.vehicle.mass_kg,
            self.vehicle.drag_coefficient,
            self.vehicle.frontal_area_m2,
            self.vehicle.battery_capacity_kwh,
            self.vehicle.motor_efficiency,
            self.route.distance_km,
            elev_gain,
            elev_loss,
            self.route.target_velocity_mps * 3.6,  # km/h
            self.environment.temperature_c,
            self.environment.wind_speed_mps,
            self.environment.wind_direction_deg,
            50.0,  # humidity (placeholder)
            self.driver_aggression,
            0.75,  # regen preference (default)
            1.0,   # aux power factor (default)
            self.initial_soc
        ], dtype=np.float32)


@dataclass
class ConfidenceInfo:
    """Detailed confidence information."""
    score: float  # 0-1
    interpretation: str  # "HIGH", "MEDIUM", "LOW"
    model_uncertainty: float
    data_quality: float
    is_in_distribution: bool
    recommendations: List[str] = field(default_factory=list)
    
    @classmethod
    def from_ml_uncertainty(
        cls,
        mean: float,
        std: float,
        input_quality: float = 1.0
    ) -> 'ConfidenceInfo':
        """Create confidence info from ML prediction uncertainty."""
        # Coefficient of variation
        cv = std / max(abs(mean), 0.01)
        
        # Base confidence from uncertainty
        model_uncertainty = math.exp(-cv)  # Higher uncertainty = lower confidence
        
        # Overall score
        score = 0.6 * model_uncertainty + 0.4 * input_quality
        
        # Interpretation
        if score >= 0.75:
            interpretation = "HIGH"
        elif score >= 0.5:
            interpretation = "MEDIUM"
        else:
            interpretation = "LOW"
        
        recommendations = []
        if score < 0.5:
            recommendations.append("Consider using physics simulation for accuracy")
        if cv > 0.3:
            recommendations.append("High model uncertainty detected")
        
        return cls(
            score=score,
            interpretation=interpretation,
            model_uncertainty=model_uncertainty,
            data_quality=input_quality,
            is_in_distribution=cv < 0.5,
            recommendations=recommendations
        )


@dataclass
class HybridPredictionResult:
    """Result from hybrid prediction system."""
    energy_kwh: float
    duration_minutes: float
    final_soc: float
    avg_speed_kmh: float
    confidence: ConfidenceInfo
    method_used: PredictionMethod
    execution_time_ms: float
    physics_result: Optional[SimulationResult] = None
    ml_prediction: Optional[float] = None
    ml_uncertainty: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "energy_kwh": self.energy_kwh,
            "duration_minutes": self.duration_minutes,
            "final_soc": self.final_soc,
            "avg_speed_kmh": self.avg_speed_kmh,
            "confidence": {
                "score": self.confidence.score,
                "interpretation": self.confidence.interpretation,
                "model_uncertainty": self.confidence.model_uncertainty,
                "data_quality": self.confidence.data_quality,
                "in_distribution": self.confidence.is_in_distribution
            },
            "method_used": self.method_used.value,
            "execution_time_ms": self.execution_time_ms,
            "recommendations": self.confidence.recommendations
        }


class HybridPredictor:
    """
    Intelligent prediction routing between ML and physics.
    
    Logic:
        1. Check if scenario is in-distribution for ML
        2. If yes and confidence > threshold: use ML (fast)
        3. If no or low confidence: use physics (accurate)
        4. Periodically cross-validate ML against physics
    """
    
    def __init__(
        self,
        ml_model: Optional[Any] = None,
        physics_engine: Optional[IntegratedSimulationEngine] = None,
        confidence_threshold: float = 0.75,
        validation_ratio: float = 0.05  # Cross-validate 5% of ML predictions
    ):
        """
        Initialize hybrid predictor.
        
        Args:
            ml_model: Trained EnergyPredictorNetwork (optional)
            physics_engine: Physics simulation engine (created if None)
            confidence_threshold: Minimum confidence for ML prediction
            validation_ratio: Fraction of ML predictions to cross-validate
        """
        self.ml_model = ml_model
        self.physics_engine = physics_engine or IntegratedSimulationEngine()
        self.confidence_threshold = confidence_threshold
        self.validation_ratio = validation_ratio
        
        # Track validation results
        self.validation_history: List[Dict[str, float]] = []
        self.prediction_count = 0
        
        # Device for ML inference
        if TORCH_AVAILABLE and ml_model is not None:
            self.device = next(ml_model.parameters()).device
        else:
            self.device = None
    
    def load_ml_model(self, model_path: str):
        """Load ML model from checkpoint."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ML model")
        
        from app.ml.models.energy_predictor import EnergyPredictorTrainer
        
        self.ml_model = EnergyPredictorTrainer.load_checkpoint(model_path)
        self.device = next(self.ml_model.parameters()).device
    
    def _predict_ml(
        self,
        scenario: ScenarioInput,
        n_mc_samples: int = 30
    ) -> Tuple[float, float, ConfidenceInfo]:
        """
        Make ML prediction with uncertainty.
        
        Returns:
            (predicted_energy, uncertainty, confidence_info)
        """
        if self.ml_model is None:
            raise ValueError("ML model not loaded")
        
        # Get features
        features = scenario.to_feature_array()
        x = torch.tensor(features).unsqueeze(0).to(self.device)
        
        # Monte Carlo prediction
        results = self.ml_model.predict_with_uncertainty(x, n_mc_samples)
        
        mean = results['mean'].item()
        std = results['std'].item()
        
        # Create confidence info
        confidence = ConfidenceInfo.from_ml_uncertainty(mean, std)
        
        return mean, std, confidence
    
    def _predict_physics(
        self,
        scenario: ScenarioInput
    ) -> SimulationResult:
        """Run full physics simulation."""
        return self.physics_engine.simulate(
            vehicle_params=scenario.vehicle,
            route_params=scenario.route,
            environment_params=scenario.environment,
            dt=1.0
        )
    
    def _should_cross_validate(self) -> bool:
        """Decide if we should cross-validate this prediction."""
        import random
        return random.random() < self.validation_ratio
    
    def _cross_validate(
        self,
        ml_prediction: float,
        physics_result: SimulationResult
    ) -> Dict[str, float]:
        """Compare ML prediction to physics result."""
        physics_energy = physics_result.total_energy_kwh
        
        abs_error = abs(ml_prediction - physics_energy)
        rel_error = abs_error / max(physics_energy, 0.01)
        
        validation = {
            'ml_prediction': ml_prediction,
            'physics_result': physics_energy,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'within_5_percent': rel_error <= 0.05,
            'within_10_percent': rel_error <= 0.10
        }
        
        self.validation_history.append(validation)
        
        # Keep only last 1000 validations
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return validation
    
    def get_validation_stats(self) -> Dict[str, float]:
        """Get statistics from cross-validation history."""
        if not self.validation_history:
            return {"n_validations": 0}
        
        rel_errors = [v['relative_error'] for v in self.validation_history]
        within_5 = sum(v['within_5_percent'] for v in self.validation_history)
        within_10 = sum(v['within_10_percent'] for v in self.validation_history)
        n = len(self.validation_history)
        
        return {
            'n_validations': n,
            'mean_relative_error': sum(rel_errors) / n,
            'max_relative_error': max(rel_errors),
            'within_5_percent': within_5 / n * 100,
            'within_10_percent': within_10 / n * 100
        }
    
    def predict_quick(self, scenario: ScenarioInput) -> HybridPredictionResult:
        """
        Fast ML-only prediction.
        
        Use when speed is priority and accuracy within 10% is acceptable.
        """
        start_time = time.time()
        
        if self.ml_model is None:
            raise ValueError("ML model not loaded. Use predict_accurate() instead.")
        
        energy, uncertainty, confidence = self._predict_ml(scenario)
        
        # Estimate other values from energy
        # Duration: rough estimate based on distance and speed
        avg_speed_mps = scenario.route.target_velocity_mps * 0.8  # 80% of target
        duration_s = (scenario.route.distance_km * 1000) / avg_speed_mps
        duration_min = duration_s / 60
        
        # Final SOC: energy used from battery
        capacity = scenario.vehicle.battery_capacity_kwh
        soc_used = (energy / capacity) * 100
        final_soc = max(0, scenario.initial_soc - soc_used)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return HybridPredictionResult(
            energy_kwh=energy,
            duration_minutes=duration_min,
            final_soc=final_soc,
            avg_speed_kmh=avg_speed_mps * 3.6,
            confidence=confidence,
            method_used=PredictionMethod.ML_ONLY,
            execution_time_ms=execution_time_ms,
            ml_prediction=energy,
            ml_uncertainty=uncertainty
        )
    
    def predict_accurate(self, scenario: ScenarioInput) -> HybridPredictionResult:
        """
        Accurate physics-only prediction.
        
        Use when accuracy is critical and time permits.
        """
        start_time = time.time()
        
        result = self._predict_physics(scenario)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Physics has high confidence by design
        confidence = ConfidenceInfo(
            score=0.95,
            interpretation="HIGH",
            model_uncertainty=0.0,
            data_quality=1.0,
            is_in_distribution=True,
            recommendations=["Physics simulation provides highest accuracy"]
        )
        
        # Calculate duration from trajectory
        duration_min = (result.trajectory[-1].time_s if result.trajectory else 0) / 60
        
        return HybridPredictionResult(
            energy_kwh=result.total_energy_kwh,
            duration_minutes=duration_min,
            final_soc=result.final_soc_percent,
            avg_speed_kmh=result.avg_velocity_mps * 3.6,
            confidence=confidence,
            method_used=PredictionMethod.PHYSICS_ONLY,
            execution_time_ms=execution_time_ms,
            physics_result=result
        )
    
    def predict_hybrid(self, scenario: ScenarioInput) -> HybridPredictionResult:
        """
        Hybrid prediction with intelligent routing.
        
        Uses ML for simple scenarios, physics for complex ones,
        and periodically cross-validates.
        """
        self.prediction_count += 1
        start_time = time.time()
        
        # If no ML model, fall back to physics
        if self.ml_model is None:
            return self.predict_accurate(scenario)
        
        # Get ML prediction with confidence
        ml_energy, ml_uncertainty, ml_confidence = self._predict_ml(scenario)
        
        # Decide routing based on confidence
        use_ml = (
            ml_confidence.score >= self.confidence_threshold and
            ml_confidence.is_in_distribution
        )
        
        # Cross-validate periodically
        should_validate = self._should_cross_validate()
        
        if use_ml and not should_validate:
            # Fast path: use ML only
            duration_min = (scenario.route.distance_km * 1000) / (
                scenario.route.target_velocity_mps * 0.8
            ) / 60
            
            capacity = scenario.vehicle.battery_capacity_kwh
            soc_used = (ml_energy / capacity) * 100
            final_soc = max(0, scenario.initial_soc - soc_used)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return HybridPredictionResult(
                energy_kwh=ml_energy,
                duration_minutes=duration_min,
                final_soc=final_soc,
                avg_speed_kmh=scenario.route.target_velocity_mps * 0.8 * 3.6,
                confidence=ml_confidence,
                method_used=PredictionMethod.ML_ONLY,
                execution_time_ms=execution_time_ms,
                ml_prediction=ml_energy,
                ml_uncertainty=ml_uncertainty
            )
        
        # Run physics simulation
        physics_result = self._predict_physics(scenario)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Cross-validate if ML was attempted
        if should_validate or use_ml:
            self._cross_validate(ml_energy, physics_result)
        
        # Determine method
        if use_ml:
            method = PredictionMethod.ML_VALIDATED
            # Use ML prediction if validation passed, else physics
            validation = self.validation_history[-1] if self.validation_history else None
            if validation and validation.get('within_10_percent', False):
                energy = ml_energy
            else:
                energy = physics_result.total_energy_kwh
                method = PredictionMethod.PHYSICS_FALLBACK
                ml_confidence.recommendations.append(
                    "ML prediction differed from physics; using physics result"
                )
        else:
            method = PredictionMethod.PHYSICS_FALLBACK
            energy = physics_result.total_energy_kwh
            ml_confidence.recommendations.append(
                "Low confidence triggered physics fallback"
            )
        
        duration_min = (physics_result.trajectory[-1].time_s if physics_result.trajectory else 0) / 60
        
        return HybridPredictionResult(
            energy_kwh=energy,
            duration_minutes=duration_min,
            final_soc=physics_result.final_soc_percent,
            avg_speed_kmh=physics_result.avg_velocity_mps * 3.6,
            confidence=ml_confidence,
            method_used=method,
            execution_time_ms=execution_time_ms,
            physics_result=physics_result,
            ml_prediction=ml_energy,
            ml_uncertainty=ml_uncertainty
        )
    
    def predict(
        self,
        scenario: ScenarioInput,
        mode: str = "hybrid"
    ) -> HybridPredictionResult:
        """
        Main prediction interface.
        
        Args:
            scenario: Input scenario
            mode: "quick" (ML), "accurate" (physics), or "hybrid"
            
        Returns:
            Prediction result
        """
        if mode == "quick":
            return self.predict_quick(scenario)
        elif mode == "accurate":
            return self.predict_accurate(scenario)
        else:
            return self.predict_hybrid(scenario)
