from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import logging
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

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
    """
    Proxy to Open-Elevation API with retries and GET fallback.
    """
    body = await request.json()
    locations = body.get("locations", [])
    
    async with httpx.AsyncClient() as client:
        # Try POST first (up to 3 times)
        for attempt in range(3):
            try:
                response = await client.post(
                    "https://api.open-elevation.com/api/v1/lookup",
                    json=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=30.0
                )
                if response.status_code == 200:
                    return JSONResponse(content=response.json())
                
                logger.warning(f"Elevation API attempt {attempt+1} failed with status {response.status_code}")
                if response.status_code == 504:
                    continue # Retry on timeout
                    
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"Elevation API attempt {attempt+1} network error: {str(e)}")
                if attempt < 2:
                    continue
                
        # Fallback to GET for small requests if POST failed
        if len(locations) < 30:
            try:
                loc_str = "|".join([f"{loc['latitude']},{loc['longitude']}" for loc in locations])
                response = await client.get(
                    "https://api.open-elevation.com/api/v1/lookup",
                    params={"locations": loc_str},
                    timeout=20.0
                )
                if response.status_code == 200:
                    logger.info("Elevation API successfully fell back to GET")
                    return JSONResponse(content=response.json())
            except Exception as e:
                logger.error(f"Elevation GET fallback failed: {str(e)}")

        raise HTTPException(
            status_code=502, 
            detail="Error communicating with Open-Elevation after multiple attempts"
        )

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
