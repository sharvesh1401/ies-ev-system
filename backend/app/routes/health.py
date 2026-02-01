from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict
from app.database import get_db, get_redis_client

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Health check endpoint to verify backend, database, and redis connectivity.
    """
    status_report = {
        "status": "healthy",
        "database": "unknown",
        "redis": "unknown"
    }
    
    # Check Database
    try:
        db.execute(text("SELECT 1"))
        status_report["database"] = "connected"
    except Exception as e:
        status_report["database"] = f"error: {str(e)}"
        status_report["status"] = "unhealthy"
        
    # Check Redis
    try:
        r = get_redis_client()
        r.ping()
        status_report["redis"] = "connected"
    except Exception as e:
        status_report["redis"] = f"error: {str(e)}"
        status_report["status"] = "unhealthy"
    
    if status_report["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status_report
        )
        
    return status_report
