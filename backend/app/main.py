from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.config import settings
from app.routes import health_router, ai_router, simulation_router
from app.routes.prediction_v2 import router as prediction_v2_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for IES_EV Project"
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes.external_apis import router as external_api_router

# Include Routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])
app.include_router(simulation_router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(prediction_v2_router, prefix="/api/v2", tags=["Prediction v2"])
app.include_router(external_api_router, prefix="/api/external", tags=["External Proxies"])

# Monitoring
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    """
    Root endpoint to verify service is running.
    """
    return {
        "message": "Welcome to IES_EV API",
        "version": settings.VERSION,
        "docs": "/docs"
    }
