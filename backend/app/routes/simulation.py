from fastapi import APIRouter, HTTPException, Depends
from app.simulation import schemas
from app.simulation.integrated_engine import IntegratedSimulationEngine

router = APIRouter()

# Dependency to get engine
def get_engine():
    return IntegratedSimulationEngine()

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
    """
    try:
        result = engine.simulate(
            vehicle_params=request.vehicle,
            route_params=request.route,
            environment_params=request.environment
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
