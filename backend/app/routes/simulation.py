from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict
import time
import traceback

from app.simulation import schemas
from app.simulation.integrated_engine import IntegratedSimulationEngine

router = APIRouter()

# Global instances for performance
_engine_instance: Optional[IntegratedSimulationEngine] = None
_hybrid_predictor_instance = None


def get_engine() -> IntegratedSimulationEngine:
    """Get or create physics engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = IntegratedSimulationEngine()
    return _engine_instance


def get_hybrid_predictor():
    """Get or create hybrid predictor."""
    global _hybrid_predictor_instance
    if _hybrid_predictor_instance is None:
        try:
            from app.ml.hybrid_predictor import HybridPredictor
            _hybrid_predictor_instance = HybridPredictor(
                ml_model=None,  # Will try to load from default path
                physics_engine=get_engine()
            )
            # Try to load ML model
            try:
                _hybrid_predictor_instance.load_ml_model("models/energy_predictor.pth")
            except Exception:
                pass  # ML model optional
        except ImportError:
            _hybrid_predictor_instance = None
    return _hybrid_predictor_instance


# ============ V1 API (Legacy) ============

class SimulationRequest(schemas.BaseModel):
    vehicle: schemas.VehicleParameters
    route: schemas.RouteParameters
    environment: schemas.EnvironmentParameters


@router.post("/run", response_model=schemas.SimulationResult)
async def run_simulation(
    request: SimulationRequest,
    engine: IntegratedSimulationEngine = Depends(get_engine)
):
    """
    Run the physics simulation with provided parameters.
    [V1 API - Legacy endpoint]
    """
    try:
        result = engine.simulate(
            vehicle_params=request.vehicle,
            route_params=request.route,
            environment_params=request.environment
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# ============ V2 API (ML-Enhanced) ============

class PredictionRequest(BaseModel):
    """Request for v2 prediction endpoints."""
    vehicle: schemas.VehicleParameters
    route: schemas.RouteParameters
    environment: schemas.EnvironmentParameters
    initial_soc: float = Field(90.0, ge=0, le=100, description="Initial battery SOC (%)")
    driver_aggression: float = Field(0.5, ge=0, le=1, description="Driver aggression factor")


class ConfidenceInfo(BaseModel):
    """Confidence information for predictions."""
    score: float = Field(..., ge=0, le=1, description="Confidence score")
    interpretation: Literal["HIGH", "MEDIUM", "LOW", "VERY_LOW"]
    model_uncertainty: Optional[float] = None
    data_quality: Optional[float] = None
    in_distribution: bool = True


class PredictionResponse(BaseModel):
    """Response for v2 prediction endpoints."""
    energy_kwh: float = Field(..., description="Predicted energy consumption (kWh)")
    duration_minutes: float = Field(..., description="Estimated trip duration (minutes)")
    final_soc: float = Field(..., description="Predicted final SOC (%)")
    avg_speed_kmh: float = Field(..., description="Average speed (km/h)")
    confidence: ConfidenceInfo
    method_used: Literal["ml", "physics", "ml_validated", "physics_fallback"]
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    recommendations: List[str] = Field(default_factory=list)


@router.post("/v2/predict-quick", response_model=PredictionResponse)
async def predict_quick(request: PredictionRequest):
    """
    Fast ML-only prediction (~100ms).
    
    Use when speed is priority. Accuracy typically within 10% of physics.
    Requires trained ML model.
    """
    start_time = time.time()
    
    predictor = get_hybrid_predictor()
    if predictor is None or predictor.ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Use /v2/predict-accurate instead."
        )
    
    try:
        from app.ml.hybrid_predictor import ScenarioInput
        
        scenario = ScenarioInput(
            vehicle=request.vehicle,
            route=request.route,
            environment=request.environment,
            initial_soc=request.initial_soc,
            driver_aggression=request.driver_aggression
        )
        
        result = predictor.predict_quick(scenario)
        
        return PredictionResponse(
            energy_kwh=result.energy_kwh,
            duration_minutes=result.duration_minutes,
            final_soc=result.final_soc,
            avg_speed_kmh=result.avg_speed_kmh,
            confidence=ConfidenceInfo(
                score=result.confidence.score,
                interpretation=result.confidence.interpretation,
                model_uncertainty=result.confidence.model_uncertainty,
                data_quality=result.confidence.data_quality,
                in_distribution=result.confidence.is_in_distribution
            ),
            method_used=result.method_used.value,
            execution_time_ms=result.execution_time_ms,
            recommendations=result.confidence.recommendations
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v2/predict-accurate", response_model=PredictionResponse)
async def predict_accurate(request: PredictionRequest):
    """
    Accurate physics-only prediction (~2000ms).
    
    Use when accuracy is critical. Full physics simulation.
    """
    start_time = time.time()
    
    predictor = get_hybrid_predictor()
    if predictor is None:
        # Fallback to direct physics
        engine = get_engine()
        try:
            result = engine.simulate(
                vehicle_params=request.vehicle,
                route_params=request.route,
                environment_params=request.environment
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            duration_min = (result.trajectory[-1].time_s if result.trajectory else 0) / 60
            
            return PredictionResponse(
                energy_kwh=result.total_energy_kwh,
                duration_minutes=duration_min,
                final_soc=result.final_soc_percent,
                avg_speed_kmh=result.avg_velocity_mps * 3.6,
                confidence=ConfidenceInfo(
                    score=0.95,
                    interpretation="HIGH",
                    in_distribution=True
                ),
                method_used="physics",
                execution_time_ms=execution_time_ms,
                recommendations=["Physics simulation provides highest accuracy"]
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
    
    try:
        from app.ml.hybrid_predictor import ScenarioInput
        
        scenario = ScenarioInput(
            vehicle=request.vehicle,
            route=request.route,
            environment=request.environment,
            initial_soc=request.initial_soc,
            driver_aggression=request.driver_aggression
        )
        
        result = predictor.predict_accurate(scenario)
        
        return PredictionResponse(
            energy_kwh=result.energy_kwh,
            duration_minutes=result.duration_minutes,
            final_soc=result.final_soc,
            avg_speed_kmh=result.avg_speed_kmh,
            confidence=ConfidenceInfo(
                score=result.confidence.score,
                interpretation=result.confidence.interpretation,
                in_distribution=True
            ),
            method_used=result.method_used.value,
            execution_time_ms=result.execution_time_ms,
            recommendations=result.confidence.recommendations
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v2/predict-hybrid", response_model=PredictionResponse)
async def predict_hybrid(request: PredictionRequest):
    """
    Hybrid prediction with intelligent routing (100-2000ms).
    
    Uses ML for simple high-confidence scenarios, physics for complex ones.
    Best balance of speed and accuracy.
    """
    predictor = get_hybrid_predictor()
    if predictor is None:
        # Fallback to physics only
        return await predict_accurate(request)
    
    try:
        from app.ml.hybrid_predictor import ScenarioInput
        
        scenario = ScenarioInput(
            vehicle=request.vehicle,
            route=request.route,
            environment=request.environment,
            initial_soc=request.initial_soc,
            driver_aggression=request.driver_aggression
        )
        
        result = predictor.predict_hybrid(scenario)
        
        return PredictionResponse(
            energy_kwh=result.energy_kwh,
            duration_minutes=result.duration_minutes,
            final_soc=result.final_soc,
            avg_speed_kmh=result.avg_speed_kmh,
            confidence=ConfidenceInfo(
                score=result.confidence.score,
                interpretation=result.confidence.interpretation,
                model_uncertainty=result.confidence.model_uncertainty,
                data_quality=result.confidence.data_quality,
                in_distribution=result.confidence.is_in_distribution
            ),
            method_used=result.method_used.value,
            execution_time_ms=result.execution_time_ms,
            recommendations=result.confidence.recommendations
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v2/validation-stats")
async def get_validation_stats():
    """
    Get ML model validation statistics.
    
    Returns cross-validation metrics from hybrid predictor.
    """
    predictor = get_hybrid_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Hybrid predictor not available")
    
    return {
        "ml_model_loaded": predictor.ml_model is not None,
        "validation_stats": predictor.get_validation_stats(),
        "confidence_threshold": predictor.confidence_threshold,
        "prediction_count": predictor.prediction_count
    }
