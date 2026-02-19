"""Tests for prediction v2 API endpoints."""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Set required env vars for Settings() if not already set
for key, val in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_DB": "test",
    "POSTGRES_HOST": "localhost",
    "REDIS_HOST": "localhost",
}.items():
    os.environ.setdefault(key, val)

# Mock database module to avoid psycopg2 dependency
_db_mock = MagicMock()
_db_mock.get_db = MagicMock()
_db_mock.get_redis_client = MagicMock()
sys.modules.setdefault("app.database", _db_mock)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared request payloads
# ---------------------------------------------------------------------------

VALID_VEHICLE = {
    "mass_kg": 1500,
    "drag_coefficient": 0.30,
    "frontal_area_m2": 2.2,
    "rolling_resistance_coefficient": 0.012,
    "battery_capacity_kwh": 60,
    "battery_voltage_nominal": 400,
    "battery_internal_resistance_ohm": 0.05,
}

VALID_ROUTE = {
    "distance_km": 10,
    "target_velocity_mps": 20,
}

VALID_ENVIRONMENT = {
    "temperature_c": 25,
    "wind_speed_mps": 0,
}


# ---------------------------------------------------------------------------
# POST /api/v2/predict-hybrid
# ---------------------------------------------------------------------------

class TestPredictHybrid:

    def test_constant_speed_prediction(self):
        """Should return a valid response for constant-speed prediction."""
        payload = {
            "vehicle": VALID_VEHICLE,
            "route": VALID_ROUTE,
            "environment": VALID_ENVIRONMENT,
            "initial_soc": 90,
        }

        response = client.post("/api/v2/predict-hybrid", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "energy_kwh" in data
        assert "duration_minutes" in data
        assert "final_soc" in data
        assert "confidence" in data
        assert "method_used" in data
        assert data["energy_kwh"] > 0

    def test_variable_speed_city(self):
        """Should return a valid response with city driving profile."""
        payload = {
            "vehicle": VALID_VEHICLE,
            "route": VALID_ROUTE,
            "environment": VALID_ENVIRONMENT,
            "profile_type": "city",
            "driver_style": "moderate",
        }

        response = client.post("/api/v2/predict-hybrid", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["driving_profile"] is not None
        assert data["driving_profile"]["type"] == "city"
        assert data["energy_breakdown"] is not None

    def test_confidence_structure(self):
        """Confidence response should have all expected fields."""
        payload = {
            "vehicle": VALID_VEHICLE,
            "route": VALID_ROUTE,
            "environment": VALID_ENVIRONMENT,
        }

        response = client.post("/api/v2/predict-hybrid", json=payload)

        assert response.status_code == 200
        conf = response.json()["confidence"]
        assert "score" in conf
        assert "level" in conf
        assert "interpretation" in conf
        assert conf["level"] in ("HIGH", "MEDIUM", "LOW", "VERY_LOW")


# ---------------------------------------------------------------------------
# POST /api/v2/driving-profile
# ---------------------------------------------------------------------------

class TestDrivingProfile:

    @pytest.mark.parametrize("profile_type", ["city", "highway", "mixed"])
    def test_generate_profile(self, profile_type):
        """Should generate a profile for each type."""
        payload = {
            "distance_km": 10,
            "profile_type": profile_type,
            "driver_style": "moderate",
            "seed": 42,
        }

        response = client.post("/api/v2/driving-profile", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["profile_type"] == profile_type
        assert data["total_distance_m"] == 10000.0
        assert len(data["speed_points"]) > 0

    def test_invalid_profile_type(self):
        """Should return 422 for invalid profile type."""
        payload = {
            "distance_km": 10,
            "profile_type": "spaceflight",
        }

        response = client.post("/api/v2/driving-profile", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v2/confidence-report
# ---------------------------------------------------------------------------

class TestConfidenceReport:

    def test_report_returns_200(self):
        """Should return a confidence report."""
        response = client.get("/api/v2/confidence-report")

        assert response.status_code == 200
        data = response.json()
        assert "ml_available" in data
        assert "models_status" in data
