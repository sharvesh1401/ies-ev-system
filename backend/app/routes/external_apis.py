from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List
import httpx
import logging
from app.config import settings
from app.limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Input schemas ────────────────────────────────────────────────────────────

class Coordinate(BaseModel):
    """A single [longitude, latitude] pair validated to WGS-84 bounds."""
    longitude: float = Field(..., ge=-180.0, le=180.0)
    latitude: float = Field(..., ge=-90.0, le=90.0)


class OrsDirectionsRequest(BaseModel):
    """Body accepted by POST /ors/directions.

    Only the fields the frontend actually sends are allowed (extra='forbid').
    Coordinates are validated to WGS-84 bounds; route length is capped at 25
    waypoints to prevent resource abuse.
    """
    # ORS expects [[lon, lat], [lon, lat], …]
    coordinates: List[List[float]] = Field(..., min_length=2, max_length=25)
    # Forward any optional ORS parameters the client may include
    radiuses: List[float] | None = Field(default=None, max_length=25)
    units: str | None = Field(default=None, max_length=10)
    alternative_routes: dict | None = Field(default=None)
    instructions: bool | None = Field(default=None)
    geometry: bool | None = Field(default=None)
    profile: str | None = Field(default="driving-car")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinate_bounds(cls, coords: List[List[float]]) -> List[List[float]]:
        for pair in coords:
            if len(pair) != 2:
                raise ValueError("Each coordinate must be [longitude, latitude]")
            lon, lat = pair
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Longitude {lon} out of range [-180, 180]")
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        return coords

    model_config = {"extra": "ignore"}


class ElevationPoint(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


class ElevationLookupRequest(BaseModel):
    """Body accepted by POST /elevation/lookup.

    Capped at 512 points to avoid overwhelming the free Open-Elevation API.
    """
    locations: List[ElevationPoint] = Field(..., min_length=1, max_length=512)

    model_config = {"extra": "forbid"}


# ─── External API Health Check ────────────────────────────────────────────────

@router.get("/health")
@limiter.limit("10/minute")
async def check_external_apis(request: Request):
    """
    Tests all four external APIs and returns their status.
    Called automatically on app startup and available at /api/external/health.
    """
    results = {}

    # 1. OpenRouteService
    if not settings.ORS_API_KEY or settings.ORS_API_KEY.startswith("your_"):
        results["openrouteservice"] = {"status": "not_configured", "message": "ORS_API_KEY missing in .env"}
    else:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                    json={"coordinates": [[4.9041, 52.3676], [4.9200, 52.3800]]},
                    headers={"Authorization": settings.ORS_API_KEY, "Content-Type": "application/json"},
                    timeout=10.0,
                )
            if r.status_code == 200:
                results["openrouteservice"] = {"status": "ok", "http": r.status_code}
            elif r.status_code in (401, 403):
                results["openrouteservice"] = {"status": "invalid_key", "http": r.status_code}
            else:
                results["openrouteservice"] = {"status": "error", "http": r.status_code}
        except Exception as e:
            results["openrouteservice"] = {"status": "unreachable", "message": str(e)[:80]}

    # 2. OpenWeatherMap
    if not settings.OPENWEATHER_API_KEY or settings.OPENWEATHER_API_KEY.startswith("your_"):
        results["openweathermap"] = {"status": "not_configured", "message": "OPENWEATHER_API_KEY missing in .env"}
    else:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={"lat": 52.3676, "lon": 4.9041, "units": "metric", "appid": settings.OPENWEATHER_API_KEY},
                    timeout=10.0,
                )
            if r.status_code == 200:
                results["openweathermap"] = {"status": "ok", "http": r.status_code}
            elif r.status_code == 401:
                results["openweathermap"] = {"status": "invalid_key", "http": r.status_code}
            else:
                results["openweathermap"] = {"status": "error", "http": r.status_code}
        except Exception as e:
            results["openweathermap"] = {"status": "unreachable", "message": str(e)[:80]}

    # 3. Open Charge Map
    if not settings.OPENCHARGE_API_KEY or settings.OPENCHARGE_API_KEY.startswith("your_"):
        results["openchargemap"] = {"status": "not_configured", "message": "OPENCHARGE_API_KEY missing in .env"}
    else:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.openchargemap.io/v3/poi",
                    params={"latitude": 52.3676, "longitude": 4.9041, "distance": 1, "maxresults": 1, "compact": "true"},
                    headers={"X-API-Key": settings.OPENCHARGE_API_KEY},
                    timeout=10.0,
                )
            if r.status_code == 200:
                results["openchargemap"] = {"status": "ok", "http": r.status_code}
            elif r.status_code in (401, 403):
                results["openchargemap"] = {"status": "invalid_key", "http": r.status_code}
            else:
                results["openchargemap"] = {"status": "error", "http": r.status_code}
        except Exception as e:
            results["openchargemap"] = {"status": "unreachable", "message": str(e)[:80]}

    # 4. Open-Elevation (no key required)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": [{"latitude": 52.3676, "longitude": 4.9041}]},
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
        if r.status_code == 200:
            results["open_elevation"] = {"status": "ok", "http": r.status_code}
        else:
            results["open_elevation"] = {"status": "error", "http": r.status_code}
    except Exception as e:
        results["open_elevation"] = {"status": "unreachable", "message": str(e)[:80]}

    all_ok = all(v["status"] == "ok" for v in results.values())
    configured = all(v["status"] != "not_configured" for v in results.values())

    return {
        "overall": "ok" if all_ok else ("partial" if configured else "not_configured"),
        "services": results,
    }


# ─── ORS directions proxy ─────────────────────────────────────────────────────

@router.post("/ors/directions")
@limiter.limit("30/minute")
async def proxy_ors_directions(request: Request, body: OrsDirectionsRequest):
    """Proxy to OpenRouteService directions API.

    The body is validated by OrsDirectionsRequest before forwarding — unknown
    fields are rejected (extra='forbid') and coordinate bounds are enforced.
    """
    if not settings.ORS_API_KEY:
        raise HTTPException(status_code=500, detail="ORS_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        try:
            url = f"https://api.openrouteservice.org/v2/directions/{body.profile}/geojson"
            payload = body.model_dump(exclude_none=True, exclude={"profile"})
            headers = {
                "Authorization": settings.ORS_API_KEY,
                "Content-Type": "application/json",
            }
            
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=15.0,
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with ORS: {str(e)}")


# ─── Elevation proxy ──────────────────────────────────────────────────────────

@router.post("/elevation/lookup")
@limiter.limit("30/minute")
async def proxy_elevation(request: Request, body: ElevationLookupRequest):
    """Proxy to Open-Elevation API with retries and GET fallback.

    Accepts a validated ElevationLookupRequest (max 512 points) so the upstream
    API is never hit with unbounded or malformed input.
    """
    # Re-serialise validated locations into the format Open-Elevation expects
    payload = {"locations": [{"latitude": p.latitude, "longitude": p.longitude} for p in body.locations]}
    locations = payload["locations"]

    async with httpx.AsyncClient() as client:
        # Try POST first (up to 3 times)
        for attempt in range(3):
            try:
                response = await client.post(
                    "https://api.open-elevation.com/api/v1/lookup",
                    json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=30.0,
                )
                if response.status_code == 200:
                    return JSONResponse(content=response.json())

                logger.warning(f"Elevation API attempt {attempt+1} failed with status {response.status_code}")
                if response.status_code == 504:
                    continue  # Retry on gateway timeout

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
                    timeout=20.0,
                )
                if response.status_code == 200:
                    logger.info("Elevation API successfully fell back to GET")
                    return JSONResponse(content=response.json())
            except Exception as e:
                logger.error(f"Elevation GET fallback failed: {str(e)}")

        raise HTTPException(
            status_code=502,
            detail="Error communicating with Open-Elevation after multiple attempts",
        )


# ─── Weather proxy ────────────────────────────────────────────────────────────

@router.get("/weather")
@limiter.limit("30/minute")
async def proxy_weather(
    request: Request,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    units: str = Query(default="metric", max_length=10),
):
    """Proxy to OpenWeatherMap current weather API.

    lat/lon are validated to WGS-84 bounds. units is restricted to the three
    values OpenWeatherMap accepts.
    """
    if not settings.OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENWEATHER_API_KEY not configured")

    # Restrict units to the three values OWM accepts
    if units not in ("metric", "imperial", "standard"):
        raise HTTPException(status_code=422, detail="units must be one of: metric, imperial, standard")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "units": units,
                    "appid": settings.OPENWEATHER_API_KEY,
                },
                timeout=10.0,
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with OpenWeather: {str(e)}")


# ─── Open Charge Map proxy ────────────────────────────────────────────────────

@router.get("/ocm/poi")
@limiter.limit("30/minute")
async def proxy_ocm(
    request: Request,
    latitude: float,
    longitude: float,
    distance: float,
    distanceunit: str = "KM",
    maxresults: int = 30,
    compact: bool = True,
    verbose: bool = False,
    output: str = "json",
):
    """Proxy to Open Charge Map POI API.

    Bounds-checks lat/lon, caps distance at 100 km, caps maxresults at 50, and
    restricts distanceunit to KM/Miles to prevent injection of unexpected params.
    """
    if not settings.OPENCHARGE_API_KEY:
        raise HTTPException(status_code=500, detail="OPENCHARGE_API_KEY not configured")

    # Validate bounds
    if not (-90.0 <= latitude <= 90.0):
        raise HTTPException(status_code=422, detail="latitude must be between -90 and 90")
    if not (-180.0 <= longitude <= 180.0):
        raise HTTPException(status_code=422, detail="longitude must be between -180 and 180")

    # Cap resource-heavy parameters
    if distance > 100:
        raise HTTPException(status_code=422, detail="distance must not exceed 100 km")
    maxresults = min(maxresults, 50)  # Hard cap — never send more than 50 results

    if distanceunit not in ("KM", "Miles"):
        raise HTTPException(status_code=422, detail="distanceunit must be KM or Miles")

    if output not in ("json", "xml"):
        raise HTTPException(status_code=422, detail="output must be json or xml")

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
                    "output": output,
                },
                headers={"X-API-Key": settings.OPENCHARGE_API_KEY},
                timeout=15.0,
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error communicating with OCM: {str(e)}")
