"""
Prediction API - Supports teacher/student/ONNX models
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging

from app.ml.prediction_service import predict_energy

router = APIRouter(prefix="/api/predict", tags=["Predictions"])
logger = logging.getLogger(__name__)

ModelType = Literal['onnx', 'student', 'teacher']

class PredictionRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=1000)
    speed_kmh: float = Field(90, gt=0, le=200)
    temperature_c: float = Field(25, ge=-40, le=50)
    initial_soc: float = Field(..., ge=0, le=100)
    initial_soh: float = Field(95, ge=50, le=100)
    mass_kg: Optional[float] = Field(1600)
    drag_coeff: Optional[float] = Field(0.28)
    model_type: Optional[ModelType] = Field('onnx', description="Model: onnx, student, teacher")

class PredictionResponse(BaseModel):
    energy_kwh: float
    final_soc: float
    final_soh: float
    confidence: float
    inference_time_ms: float
    model_used: str

@router.post("/energy", response_model=PredictionResponse)
async def predict_trip_energy(request: PredictionRequest):
    """Predict energy consumption using loaded ML models."""
    try:
        result = predict_energy(
            distance_km=request.distance_km,
            speed_kmh=request.speed_kmh,
            temperature_c=request.temperature_c,
            initial_soc=request.initial_soc,
            initial_soh=request.initial_soh,
            mass_kg=request.mass_kg or 1600,
            drag_coeff=request.drag_coeff or 0.28,
            model_type=request.model_type or 'onnx'
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    from app.ml.prediction_service import prediction_service
    return prediction_service.health_check()

@router.get("/models")
async def list_models():
    from app.ml.model_loader import model_loader
    return model_loader.get_model_info()
