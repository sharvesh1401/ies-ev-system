"""
Advanced Confidence Scorer for ML Predictions.

Multi-factor confidence assessment combining:
- Model uncertainty (from MC Dropout)
- Physics agreement (when available)
- Historical accuracy
- Data quality
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


@dataclass
class ConfidenceComponents:
    """Individual confidence components."""
    model_uncertainty: float  # From MC dropout variance
    physics_agreement: float  # ML vs physics comparison
    historical_accuracy: float  # Past performance on similar scenarios
    data_quality: float  # Input feature validity
    
    # Weights
    WEIGHT_MODEL_UNCERTAINTY = 0.40
    WEIGHT_PHYSICS_AGREEMENT = 0.30
    WEIGHT_HISTORICAL_ACCURACY = 0.20
    WEIGHT_DATA_QUALITY = 0.10
    
    def calculate_overall(self) -> float:
        """Calculate weighted overall confidence."""
        return (
            self.model_uncertainty * self.WEIGHT_MODEL_UNCERTAINTY +
            self.physics_agreement * self.WEIGHT_PHYSICS_AGREEMENT +
            self.historical_accuracy * self.WEIGHT_HISTORICAL_ACCURACY +
            self.data_quality * self.WEIGHT_DATA_QUALITY
        )


@dataclass
class ConfidenceResult:
    """Full confidence assessment result."""
    score: float  # 0.0 to 1.0
    interpretation: str  # "HIGH", "MEDIUM", "LOW", "VERY_LOW"
    components: ConfidenceComponents
    recommendations: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence": self.score,
            "interpretation": self.interpretation,
            "components": {
                "model_uncertainty": self.components.model_uncertainty,
                "physics_agreement": self.components.physics_agreement,
                "historical_accuracy": self.components.historical_accuracy,
                "data_quality": self.components.data_quality
            },
            "recommendations": self.recommendations,
            "flags": self.flags
        }


@dataclass
class ScenarioSignature:
    """Signature for categorizing similar scenarios."""
    distance_bucket: int  # 0-5 (short to long)
    speed_bucket: int  # 0-3 (slow to fast)
    temp_bucket: int  # 0-2 (cold, normal, hot)
    elevation_bucket: int  # 0-2 (flat, moderate, steep)
    
    def as_tuple(self) -> Tuple[int, ...]:
        return (self.distance_bucket, self.speed_bucket, 
                self.temp_bucket, self.elevation_bucket)


class MLConfidenceScorer:
    """
    Multi-factor confidence scoring for ML predictions.
    
    Combines multiple signals to assess prediction reliability:
    1. Model Uncertainty (40%): From Monte Carlo dropout variance
    2. Physics Agreement (30%): How well ML matches physics
    3. Historical Accuracy (20%): Past performance on similar scenarios
    4. Data Quality (10%): Input validity and completeness
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize scorer.
        
        Args:
            history_size: Maximum historical predictions to track
        """
        # Historical accuracy tracking by scenario signature
        self.accuracy_history: Dict[Tuple, deque] = {}
        self.history_size = history_size
        
        # Valid ranges for input features
        self.feature_ranges = {
            'mass_kg': (800, 3500),
            'drag_coefficient': (0.15, 0.50),
            'frontal_area_m2': (1.5, 4.0),
            'battery_capacity_kwh': (20, 150),
            'motor_efficiency': (0.80, 0.98),
            'distance_km': (1, 500),
            'target_speed_kmh': (10, 200),
            'temperature_c': (-30, 50),
            'wind_speed_mps': (0, 30),
            'initial_soc': (5, 100)
        }
    
    def _calculate_model_uncertainty_score(
        self,
        mean: float,
        std: float,
        aleatoric_std: float = 0.0,
        epistemic_std: float = 0.0
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence from model uncertainty.
        
        Lower relative uncertainty = higher confidence.
        """
        flags = []
        
        # Coefficient of variation
        cv = std / max(abs(mean), 0.001)
        
        # Transform to confidence score
        # cv=0 → score=1.0, cv=0.5 → score≈0.6, cv=1.0 → score≈0.37
        score = math.exp(-cv)
        
        # Flag high uncertainty
        if cv > 0.3:
            flags.append("HIGH_MODEL_UNCERTAINTY")
        if epistemic_std > 0 and epistemic_std > aleatoric_std:
            flags.append("HIGH_EPISTEMIC_UNCERTAINTY")
        
        return score, flags
    
    def _calculate_physics_agreement_score(
        self,
        ml_prediction: float,
        physics_result: Optional[float]
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence from ML-physics agreement.
        
        If no physics result, returns neutral score.
        """
        flags = []
        
        if physics_result is None:
            return 0.7, []  # Neutral when no comparison available
        
        # Relative error
        rel_error = abs(ml_prediction - physics_result) / max(physics_result, 0.001)
        
        # Transform to score
        # error=0 → score=1.0, error=0.1 → score≈0.9, error=0.3 → score≈0.74
        score = math.exp(-rel_error * 3)
        
        # Flags
        if rel_error > 0.10:
            flags.append("ML_PHYSICS_DISCREPANCY")
        if rel_error > 0.20:
            flags.append("LARGE_PREDICTION_ERROR")
        
        return score, flags
    
    def _calculate_historical_accuracy_score(
        self,
        scenario_signature: ScenarioSignature
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence from historical accuracy on similar scenarios.
        """
        flags = []
        key = scenario_signature.as_tuple()
        
        if key not in self.accuracy_history or len(self.accuracy_history[key]) < 5:
            # Not enough history, return prior
            return 0.8, ["LIMITED_HISTORICAL_DATA"]
        
        history = list(self.accuracy_history[key])
        
        # Calculate accuracy metrics
        accuracies = [h['within_10_percent'] for h in history]
        accuracy_rate = sum(accuracies) / len(accuracies)
        
        # Recent trend (last 10 vs overall)
        if len(history) >= 10:
            recent_accuracy = sum(accuracies[-10:]) / 10
            if recent_accuracy < accuracy_rate - 0.1:
                flags.append("DECLINING_ACCURACY")
        
        return accuracy_rate, flags
    
    def _calculate_data_quality_score(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence from input data quality.
        
        Checks for out-of-range values and missing data.
        """
        flags = []
        n_features = 0
        n_valid = 0
        
        for name, value in features.items():
            if name in self.feature_ranges:
                n_features += 1
                min_val, max_val = self.feature_ranges[name]
                
                if min_val <= value <= max_val:
                    n_valid += 1
                else:
                    flags.append(f"OUT_OF_RANGE_{name.upper()}")
        
        if n_features == 0:
            return 0.5, ["NO_FEATURE_VALIDATION"]
        
        score = n_valid / n_features
        
        return score, flags
    
    def _get_scenario_signature(
        self,
        features: Dict[str, float]
    ) -> ScenarioSignature:
        """Create signature for scenario categorization."""
        # Distance buckets: [0-10, 10-30, 30-70, 70-150, 150-300, 300+] km
        distance = features.get('distance_km', 50)
        if distance < 10:
            dist_bucket = 0
        elif distance < 30:
            dist_bucket = 1
        elif distance < 70:
            dist_bucket = 2
        elif distance < 150:
            dist_bucket = 3
        elif distance < 300:
            dist_bucket = 4
        else:
            dist_bucket = 5
        
        # Speed buckets: [0-40, 40-80, 80-120, 120+] km/h
        speed = features.get('target_speed_kmh', 60)
        if speed < 40:
            speed_bucket = 0
        elif speed < 80:
            speed_bucket = 1
        elif speed < 120:
            speed_bucket = 2
        else:
            speed_bucket = 3
        
        # Temperature buckets: [cold (<10), normal (10-30), hot (>30)]
        temp = features.get('temperature_c', 20)
        if temp < 10:
            temp_bucket = 0
        elif temp < 30:
            temp_bucket = 1
        else:
            temp_bucket = 2
        
        # Elevation (based on gain/loss ratio to distance)
        elev_gain = features.get('elevation_gain_m', 0)
        if distance > 0:
            elev_ratio = elev_gain / (distance * 1000) * 100  # % grade equivalent
        else:
            elev_ratio = 0
        
        if elev_ratio < 1:
            elev_bucket = 0
        elif elev_ratio < 3:
            elev_bucket = 1
        else:
            elev_bucket = 2
        
        return ScenarioSignature(dist_bucket, speed_bucket, temp_bucket, elev_bucket)
    
    def _get_interpretation(self, score: float) -> str:
        """Get human-readable interpretation."""
        if score >= 0.85:
            return "HIGH"
        elif score >= 0.70:
            return "MEDIUM"
        elif score >= 0.50:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_recommendations(
        self,
        score: float,
        flags: List[str],
        components: ConfidenceComponents
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if score >= 0.85:
            recommendations.append("✓ All confidence checks passed")
            return recommendations
        
        if "HIGH_MODEL_UNCERTAINTY" in flags:
            recommendations.append("Consider physics simulation for higher accuracy")
        
        if "ML_PHYSICS_DISCREPANCY" in flags:
            recommendations.append("ML and physics predictions differ significantly")
        
        if "DECLINING_ACCURACY" in flags:
            recommendations.append("Accuracy on similar scenarios has been declining")
        
        if any("OUT_OF_RANGE" in f for f in flags):
            recommendations.append("Some input values are outside training distribution")
        
        if components.historical_accuracy < 0.7:
            recommendations.append("Historical accuracy on similar scenarios is below 70%")
        
        if score < 0.5:
            recommendations.append("⚠️ Low confidence - use physics simulation")
        
        return recommendations
    
    def calculate_confidence(
        self,
        prediction_mean: float,
        prediction_std: float,
        features: Dict[str, float],
        physics_result: Optional[float] = None,
        aleatoric_std: float = 0.0,
        epistemic_std: float = 0.0
    ) -> ConfidenceResult:
        """
        Calculate comprehensive confidence score.
        
        Args:
            prediction_mean: ML prediction value
            prediction_std: Total uncertainty (std)
            features: Input features as dictionary
            physics_result: Optional physics simulation result for comparison
            aleatoric_std: Data uncertainty
            epistemic_std: Model uncertainty
            
        Returns:
            ConfidenceResult with score, interpretation, and recommendations
        """
        all_flags = []
        
        # 1. Model uncertainty component
        model_score, model_flags = self._calculate_model_uncertainty_score(
            prediction_mean, prediction_std, aleatoric_std, epistemic_std
        )
        all_flags.extend(model_flags)
        
        # 2. Physics agreement component
        physics_score, physics_flags = self._calculate_physics_agreement_score(
            prediction_mean, physics_result
        )
        all_flags.extend(physics_flags)
        
        # 3. Historical accuracy component
        signature = self._get_scenario_signature(features)
        history_score, history_flags = self._calculate_historical_accuracy_score(signature)
        all_flags.extend(history_flags)
        
        # 4. Data quality component
        quality_score, quality_flags = self._calculate_data_quality_score(features)
        all_flags.extend(quality_flags)
        
        # Combine components
        components = ConfidenceComponents(
            model_uncertainty=model_score,
            physics_agreement=physics_score,
            historical_accuracy=history_score,
            data_quality=quality_score
        )
        
        overall_score = components.calculate_overall()
        interpretation = self._get_interpretation(overall_score)
        recommendations = self._generate_recommendations(overall_score, all_flags, components)
        
        return ConfidenceResult(
            score=overall_score,
            interpretation=interpretation,
            components=components,
            recommendations=recommendations,
            flags=all_flags
        )
    
    def record_validation(
        self,
        features: Dict[str, float],
        ml_prediction: float,
        actual_value: float
    ):
        """
        Record a validation result for historical tracking.
        
        Args:
            features: Input features
            ml_prediction: What ML predicted
            actual_value: Ground truth (from physics or real data)
        """
        signature = self._get_scenario_signature(features)
        key = signature.as_tuple()
        
        if key not in self.accuracy_history:
            self.accuracy_history[key] = deque(maxlen=self.history_size)
        
        rel_error = abs(ml_prediction - actual_value) / max(actual_value, 0.001)
        
        self.accuracy_history[key].append({
            'ml_prediction': ml_prediction,
            'actual': actual_value,
            'relative_error': rel_error,
            'within_5_percent': rel_error <= 0.05,
            'within_10_percent': rel_error <= 0.10
        })
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics across all scenarios."""
        all_errors = []
        total_count = 0
        within_5 = 0
        within_10 = 0
        
        for history in self.accuracy_history.values():
            for entry in history:
                all_errors.append(entry['relative_error'])
                total_count += 1
                if entry['within_5_percent']:
                    within_5 += 1
                if entry['within_10_percent']:
                    within_10 += 1
        
        if not all_errors:
            return {"n_samples": 0}
        
        return {
            "n_samples": total_count,
            "mean_relative_error": sum(all_errors) / len(all_errors),
            "max_relative_error": max(all_errors),
            "within_5_percent": within_5 / total_count * 100,
            "within_10_percent": within_10 / total_count * 100,
            "n_scenario_types": len(self.accuracy_history)
        }
