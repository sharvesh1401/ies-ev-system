"""Tests for the hybrid predictor."""

import pytest
from unittest.mock import MagicMock, patch

from app.simulation.schemas import (
    VehicleParameters, RouteParameters, EnvironmentParameters
)


@pytest.fixture
def vehicle():
    return VehicleParameters(
        mass_kg=1500,
        drag_coefficient=0.30,
        frontal_area_m2=2.2,
        rolling_resistance_coefficient=0.012,
        battery_capacity_kwh=60,
        battery_voltage_nominal=400,
        battery_internal_resistance_ohm=0.05,
    )


@pytest.fixture
def route():
    return RouteParameters(distance_km=20, target_velocity_mps=25)


@pytest.fixture
def environment():
    return EnvironmentParameters(temperature_c=25, wind_speed_mps=0)


class TestHybridPredictorPhysicsFallback:
    """Test hybrid predictor when ML models are unavailable."""

    def test_physics_only_prediction(self, vehicle, route, environment):
        """Should fall back to physics when no ML model loaded."""
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput
        from app.simulation.integrated_engine import IntegratedSimulationEngine

        engine = IntegratedSimulationEngine()
        predictor = HybridPredictor(ml_model=None, physics_engine=engine)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
            initial_soc=90.0,
            driver_aggression=0.5,
        )

        result = predictor.predict_accurate(scenario)

        assert result.energy_kwh > 0
        assert 0 <= result.final_soc <= 100
        assert result.duration_minutes > 0
        assert result.confidence.score > 0

    def test_predict_quick_without_ml_raises(self, vehicle, route, environment):
        """predict_quick should raise when ML model unavailable."""
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput

        predictor = HybridPredictor(ml_model=None, physics_engine=None)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
        )

        # With no ML model, predict_quick should raise ValueError
        with pytest.raises(ValueError, match="ML model not loaded"):
            predictor.predict(scenario, mode="quick")

    def test_hybrid_auto_routes_to_physics(self, vehicle, route, environment):
        """Without ML, hybrid should route to physics."""
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput
        from app.simulation.integrated_engine import IntegratedSimulationEngine

        engine = IntegratedSimulationEngine()
        predictor = HybridPredictor(ml_model=None, physics_engine=engine)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
        )

        result = predictor.predict(scenario, mode="hybrid")
        assert result.energy_kwh > 0


class TestPredictionResultStructure:
    """Test that prediction results have correct structure."""

    def test_result_has_all_fields(self, vehicle, route, environment):
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput
        from app.simulation.integrated_engine import IntegratedSimulationEngine

        engine = IntegratedSimulationEngine()
        predictor = HybridPredictor(ml_model=None, physics_engine=engine)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
        )

        result = predictor.predict_accurate(scenario)

        # Check all required fields
        assert hasattr(result, "energy_kwh")
        assert hasattr(result, "duration_minutes")
        assert hasattr(result, "final_soc")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method_used")
        assert hasattr(result, "execution_time_ms")

    def test_to_dict_serialisation(self, vehicle, route, environment):
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput
        from app.simulation.integrated_engine import IntegratedSimulationEngine

        engine = IntegratedSimulationEngine()
        predictor = HybridPredictor(ml_model=None, physics_engine=engine)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
        )

        result = predictor.predict_accurate(scenario)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "energy_kwh" in d
        assert "confidence" in d

    def test_energy_bounds_reasonable(self, vehicle, route, environment):
        from app.ml.hybrid_predictor import HybridPredictor, ScenarioInput
        from app.simulation.integrated_engine import IntegratedSimulationEngine

        engine = IntegratedSimulationEngine()
        predictor = HybridPredictor(ml_model=None, physics_engine=engine)

        scenario = ScenarioInput(
            vehicle=vehicle,
            route=route,
            environment=environment,
        )

        result = predictor.predict_accurate(scenario)
        # 20 km at reasonable efficiency: 1-10 kWh expected
        assert 0.5 <= result.energy_kwh <= 20.0
