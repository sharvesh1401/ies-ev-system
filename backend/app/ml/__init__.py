"""
Machine Learning Module for IES-EV System.

Provides ML-based energy prediction with uncertainty quantification
and intelligent hybrid prediction routing.
"""

from typing import TYPE_CHECKING

# Lazy imports for better startup performance
if TYPE_CHECKING:
    from app.ml.data_generator import PhysicsDataGenerator, ScenarioFeatures, ScenarioLabels
    from app.ml.models.energy_predictor import EnergyPredictorNetwork, EnergyPredictorTrainer
    from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput, HybridPredictionResult
    from app.ml.confidence_scorer import MLConfidenceScorer, ConfidenceResult
    from app.ml.validation import MLValidationSuite, ValidationReport


def get_data_generator():
    """Get the PhysicsDataGenerator class."""
    from app.ml.data_generator import PhysicsDataGenerator
    return PhysicsDataGenerator


def get_hybrid_predictor():
    """Get the HybridPredictor class."""
    from app.ml.hybrid_predictor import HybridPredictor
    return HybridPredictor


def get_confidence_scorer():
    """Get the MLConfidenceScorer class."""
    from app.ml.confidence_scorer import MLConfidenceScorer
    return MLConfidenceScorer


def get_validation_suite():
    """Get the MLValidationSuite class."""
    from app.ml.validation import MLValidationSuite
    return MLValidationSuite


def get_model_loader():
    """Get the ModelLoader class."""
    from app.ml.model_loader import ModelLoader
    return ModelLoader


__all__ = [
    # Data Generation
    'get_data_generator',
    
    # Hybrid Prediction
    'get_hybrid_predictor',
    
    # Confidence
    'get_confidence_scorer',
    
    # Validation
    'get_validation_suite',
    
    # Model Loading
    'get_model_loader',
]
