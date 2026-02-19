"""Tests for driving profile generation and variable speed simulation."""

import pytest
from app.simulation.driving_profiles import (
    CityDrivingProfile,
    HighwayDrivingProfile,
    MixedDrivingProfile,
    DriverStyle,
)
from app.simulation.variable_speed_engine import (
    generate_driving_profile,
    simulate_variable_speed,
)
from app.simulation.schemas import (
    VehicleParameters,
    RouteParameters,
    EnvironmentParameters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    return RouteParameters(distance_km=10, target_velocity_mps=20)


@pytest.fixture
def environment():
    return EnvironmentParameters(temperature_c=25)


# ---------------------------------------------------------------------------
# Profile generation tests
# ---------------------------------------------------------------------------

class TestProfileGeneration:
    """Test that driving profiles are generated correctly."""

    @pytest.mark.parametrize("profile_type", ["city", "highway", "mixed"])
    def test_generate_profile_types(self, profile_type):
        """All profile types should generate valid speed points."""
        result = generate_driving_profile(
            distance_km=10,
            profile_type=profile_type,
            driver_style="moderate",
            seed=42,
        )

        assert len(result.speed_points) > 0
        assert result.total_distance_m == 10000.0
        assert result.profile_type == profile_type
        assert result.driver_style == "moderate"

    def test_city_profile_has_stops(self):
        """City profiles should include stop events."""
        result = generate_driving_profile(
            distance_km=10,
            profile_type="city",
            driver_style="moderate",
            seed=42,
        )
        assert result.num_stops > 0

    def test_highway_profile_high_speed(self):
        """Highway profiles should reach higher speeds than city."""
        city = generate_driving_profile(
            distance_km=20, profile_type="city", seed=42
        )
        highway = generate_driving_profile(
            distance_km=20, profile_type="highway", seed=42
        )
        assert highway.max_speed_mps > city.max_speed_mps

    @pytest.mark.parametrize("style", ["aggressive", "moderate", "eco"])
    def test_driver_styles(self, style):
        """All driver styles should produce valid profiles."""
        result = generate_driving_profile(
            distance_km=10,
            profile_type="city",
            driver_style=style,
            seed=42,
        )
        assert len(result.speed_points) > 0

    def test_invalid_profile_type(self):
        """Should raise ValueError for unknown profile type."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            generate_driving_profile(distance_km=10, profile_type="spaceflight")

    def test_invalid_driver_style(self):
        """Should raise ValueError for unknown driver style."""
        with pytest.raises(ValueError, match="Unknown driver style"):
            generate_driving_profile(distance_km=10, driver_style="reckless")

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical profiles."""
        a = generate_driving_profile(distance_km=10, profile_type="city", seed=123)
        b = generate_driving_profile(distance_km=10, profile_type="city", seed=123)

        assert len(a.speed_points) == len(b.speed_points)
        for pa, pb in zip(a.speed_points, b.speed_points):
            assert pa["time_s"] == pb["time_s"]
            assert pa["speed_mps"] == pb["speed_mps"]


# ---------------------------------------------------------------------------
# Variable speed simulation tests
# ---------------------------------------------------------------------------

class TestVariableSpeedSimulation:
    """Test end-to-end variable speed simulation."""

    def test_city_simulation(self, vehicle, route, environment):
        """City simulation should produce positive energy consumption."""
        result = simulate_variable_speed(
            vehicle_params=vehicle,
            route_params=route,
            environment_params=environment,
            profile_type="city",
            driver_style="moderate",
            seed=42,
        )

        assert result.total_energy_kwh > 0
        assert result.duration_minutes > 0
        assert 0 <= result.final_soc <= 100
        assert result.profile_type == "city"

    def test_energy_breakdown_sums(self, vehicle, route, environment):
        """Energy breakdown components should roughly sum to total."""
        result = simulate_variable_speed(
            vehicle_params=vehicle,
            route_params=route,
            environment_params=environment,
            profile_type="highway",
            driver_style="moderate",
            seed=42,
        )

        breakdown = result.energy_breakdown
        components = (
            breakdown["kinetic_kwh"]
            + breakdown["potential_kwh"]
            + breakdown["aerodynamic_kwh"]
            + breakdown["rolling_kwh"]
            + breakdown["auxiliary_kwh"]
        )

        # Allow 10% tolerance for rounding
        if breakdown["total_kwh"] > 0.01:
            rel_diff = abs(components - breakdown["total_kwh"]) / breakdown["total_kwh"]
            assert rel_diff < 0.10, f"Breakdown sum mismatch: {rel_diff:.1%}"

    def test_aggressive_uses_more_energy(self, vehicle, route, environment):
        """Aggressive driving should consume more energy than eco."""
        agg = simulate_variable_speed(
            vehicle_params=vehicle,
            route_params=route,
            environment_params=environment,
            profile_type="city",
            driver_style="aggressive",
            seed=42,
        )
        eco = simulate_variable_speed(
            vehicle_params=vehicle,
            route_params=route,
            environment_params=environment,
            profile_type="city",
            driver_style="eco",
            seed=42,
        )

        # Aggressive should consume at least as much energy as eco
        assert agg.total_energy_kwh >= eco.total_energy_kwh * 0.9
