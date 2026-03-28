from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from app.config import settings

router = APIRouter()

@router.post("/ors/directions")
async def proxy_ors_directions(request: Request):
    if not settings.ORS_API_KEY:
        raise HTTPException(status_code=500, detail="ORS_API_KEY not configured")
    
    body = await request.json()
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                json=body,
                headers={
                    "Authorization": settings.ORS_API_KEY,
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with ORS: {str(e)}")

@router.post("/elevation/lookup")
async def proxy_elevation(request: Request):
    # Elevation API doesn't strictly need a key, but proxied to avoid CORS and keep requests unified
    body = await request.json()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=15.0
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with Open-Elevation: {str(e)}")

@router.get("/weather")
async def proxy_weather(lat: float, lon: float, units: str = "metric"):
    if not settings.OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENWEATHER_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "units": units,
                    "appid": settings.OPENWEATHER_API_KEY
                },
                timeout=10.0
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with OpenWeather: {str(e)}")

@router.get("/ocm/poi")
async def proxy_ocm(
    latitude: float, 
    longitude: float, 
    distance: float, 
    distanceunit: str = "KM",
    maxresults: int = 30,
    compact: bool = True,
    verbose: bool = False,
    output: str = "json"
):
    if not settings.OPENCHARGE_API_KEY:
        raise HTTPException(status_code=500, detail="OPENCHARGE_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.openchargemap.io/v3/poi",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance": distance,
                    "distanceunit": distanceunit,
                    "maxresults": maxresults,
                    "compact": str(compact).lower(),
                    "verbose": str(verbose).lower(),
                    "output": output
                },
                headers={"X-API-Key": settings.OPENCHARGE_API_KEY},
                timeout=15.0
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with OCM: {str(e)}")
