from app.simulation import vehicle_dynamics
from app.simulation.schemas import VehicleParameters, EnvironmentParameters, RouteParameters
import pytest
import math

class TestVehicleDynamics:
    
    @pytest.fixture
    def basic_vehicle(self):
        return VehicleParameters(
            mass_kg=1500,
            drag_coefficient=0.3,
            frontal_area_m2=2.2,
            rolling_resistance_coefficient=0.01,
            battery_capacity_kwh=60,
            battery_voltage_nominal=400,
            battery_internal_resistance_ohm=0.05
        )
        
    @pytest.fixture
    def basic_env(self):
        return EnvironmentParameters(
            temperature_c=25,
            wind_speed_mps=0,
            road_grade_percent=0
        )

    def test_constant_velocity_forces(self, basic_vehicle, basic_env):
        """
        At constant velocity on flat road, Tractive Force should equal Resistive Forces.
        Resistive = Aero + Roll
        """
        engine = vehicle_dynamics.VehicleDynamicsEngine()
        target_v = 20.0 # m/s
        engine.v = target_v # Set current speed
        
        # Run one step
        res = engine.simulate_step(
            vehicle=basic_vehicle,
            environment=basic_env,
            target_velocity=target_v,
            grade_rad=0.0,
            dt=0.1
        )
        
        # Check acceleration is near zero (steady state with P-controller might have small error)
        # But we want to verify forces calculation primarily.
        
        # Expected Aero: 0.5 * 1.225 * 0.3 * 2.2 * 20^2 = 161.7 N
        expected_aero = 0.5 * 1.225 * 0.3 * 2.2 * (20**2)
        assert abs(res['F_aero'] - expected_aero) < 1.0
        
        # Expected Roll: 1500 * 9.81 * 0.01 * 1 = 147.15 N
        expected_roll = 1500 * 9.81 * 0.01
        assert abs(res['F_roll'] - expected_roll) < 1.0
        
        # F_grade should be 0
        assert abs(res['F_grade']) < 0.1

    def test_hill_climbing(self, basic_vehicle, basic_env):
        """
        Check F_grade component.
        """
        engine = vehicle_dynamics.VehicleDynamicsEngine()
        grade_pct = 10.0 # 10% grade
        grade_rad = math.atan(0.1)
        
        res = engine.simulate_step(
            vehicle=basic_vehicle,
            environment=basic_env,
            target_velocity=0,
            grade_rad=grade_rad,
            dt=0.1
        )
        
        # F_grade = mg sin(theta)
        expected_grade = 1500 * 9.81 * math.sin(grade_rad)
        assert abs(res['F_grade'] - expected_grade) < 1.0
