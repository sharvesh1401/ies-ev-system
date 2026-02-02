from typing import Dict, Any, List
from datetime import datetime
import math

# Import all components
from app.simulation.schemas import (
    VehicleParameters, RouteParameters, EnvironmentParameters,
    SimulationResult, SimulationStep, ValidationReport, ConfidenceScore
)
from app.simulation.vehicle_dynamics import VehicleDynamicsEngine
from app.simulation.battery_model import BatteryModel
from app.simulation.environment import EnvironmentSimulator
from app.simulation.energy_calculator import EnergyCalculator
from app.simulation.validation import PhysicsValidationSuite
from app.simulation.confidence_scorer import ConfidenceScorer
from app.simulation.telemetry import TelemetryGenerator

class IntegratedSimulationEngine:
    """
    Complete simulation engine integrating all components.
    """
    
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamicsEngine()
        self.environment = EnvironmentSimulator()
        self.validator = PhysicsValidationSuite()
        self.confidence_scorer = ConfidenceScorer()
        self.telemetry = TelemetryGenerator()
        
    def simulate(
        self,
        vehicle_params: VehicleParameters,
        route_params: RouteParameters,
        environment_params: EnvironmentParameters,
        dt: float = 1.0
    ) -> SimulationResult:
        """
        Run complete simulation.
        """
        # 1. Initialize Objects
        battery = BatteryModel(
            capacity_kwh=vehicle_params.battery_capacity_kwh,
            voltage_nominal=vehicle_params.battery_voltage_nominal,
            internal_resistance_ohm=vehicle_params.battery_internal_resistance_ohm
        )
        
        energy_calc = EnergyCalculator(auxiliary_power_kw=vehicle_params.auxiliary_power_kw)
        
        # 2. Extract Route Info
        # If elevation profile missing, use constant grade
        # If provided, environment simulator handles it?
        # route_params has elevation_profile.
        
        # 3. Simulation Loop
        trajectory: List[SimulationStep] = []
        
        # Running variables
        current_v = 0.0 # Start from stop
        current_x = 0.0
        current_t = 0.0
        current_soc = 90.0 # Start SOC assumption? Or add to params. Let's assume 90%
        current_temp = environment_params.temperature_c
        
        # We need to set initial state in physics engine
        self.vehicle_dynamics.v = current_v
        self.vehicle_dynamics.x = current_x
        self.vehicle_dynamics.t = current_t
        
        # Determine total distance
        total_dist_m = route_params.distance_km * 1000.0
        
        # Loop condition: distance < total AND time < max_time
        max_time = 3600 * 10 # 10 hours limit
        
        while current_x < total_dist_m and current_t < max_time:
            # A. Get Environment
            # Simple assumption: Grade is constant or looked up
            # For phase 0/1 simplicity, if profile provided use it, else constant
            if route_params.elevation_profile:
                grade_pct = self.environment.get_grade_at_distance(current_x, self.environment.extract_road_grade(route_params.elevation_profile))
            else:
                grade_pct = environment_params.road_grade_percent
                
            grade_rad = math.atan(grade_pct / 100.0)
            
            # Elevation (integrate grade or look up)
            # Simple integration:
            elevation = route_params.initial_elevation_m + (current_x * math.sin(grade_rad)) # Approximation
            
            # B. Vehicle Dynamics Step
            vd_res = self.vehicle_dynamics.simulate_step(
                vehicle_params, 
                environment_params, 
                route_params.target_velocity_mps,
                grade_rad,
                dt
            )
            
            # C. Energy & Power
            pf_res = energy_calc.calculate_power_flow(
                force_tractive=vd_res['F_tractive'],
                velocity_mps=vd_res['velocity'],
                vehicle=vehicle_params,
                dt=dt
            )
            
            # D. Battery Step
            bat_res = battery.update_state(
                current_soc=current_soc,
                current_temp_c=current_temp,
                power_demand_kw=pf_res['power_battery_kw'],
                ambient_temp_c=environment_params.temperature_c,
                dt=dt
            )
            
            # Create Step Object
            step = SimulationStep(
                time_s=vd_res['time'],
                distance_m=vd_res['position'],
                velocity_mps=vd_res['velocity'],
                acceleration_mps2=vd_res['acceleration'],
                elevation_m=elevation, 
                grade_percent=grade_pct,
                
                force_tractive_n=vd_res['F_tractive'],
                force_aero_n=vd_res['F_aero'],
                force_roll_n=vd_res['F_roll'],
                force_grade_n=vd_res['F_grade'],
                
                power_motor_kw=pf_res['power_wheels_kw'], # Motor output mechanical power
                power_battery_kw=pf_res['power_battery_kw'],
                energy_step_kwh=pf_res['energy_step_kwh'],
                
                soc_percent=bat_res['soc'],
                voltage_v=bat_res['voltage'],
                current_a=bat_res['current'],
                temperature_c=bat_res['temperature']
            )
            
            trajectory.append(step)
            
            # Update loop variables
            current_v = vd_res['velocity']
            current_x = vd_res['position']
            current_t = vd_res['time']
            current_soc = bat_res['soc']
            current_temp = bat_res['temperature']
            
            # Stop if SOC depleted
            if current_soc <= 0:
                break
                
        # 4. Post-Process
        energy_breakdown = energy_calc.calculate_energy_breakdown(trajectory, vehicle_params)
        
        # Calculate totals
        total_energy_kwh = sum(step.energy_step_kwh for step in trajectory)
        avg_velocity = sum(step.velocity_mps for step in trajectory) / len(trajectory) if trajectory else 0
        final_soc = trajectory[-1].soc_percent if trajectory else 90.0
        
        # 5. Validation & Confidence
        # Construct result partially to pass to validation
        partial_result = SimulationResult(
            trajectory=trajectory,
            total_energy_kwh=total_energy_kwh,
            final_soc_percent=final_soc,
            avg_velocity_mps=avg_velocity,
            energy_breakdown=energy_breakdown, 
            validation_report=ValidationReport(overall_score=0, tests_passed=0, total_tests=0, results=[], interpretation="Pending"), # Placeholder
            confidence_score=ConfidenceScore(overall=0, physics_validation=0, uncertainty=0, historical_accuracy=0, interpretation="Pending", recommendations=[]), # Placeholder
            metadata={'vehicle': vehicle_params.model_dump(), 'timestamp': datetime.now().isoformat()}
        )
        
        validation_report = self.validator.run_all_tests(partial_result)
        
        confidence = self.confidence_scorer.calculate_confidence(
            partial_result, # Note: this has dummy report, but scorer logic reads partial_result.validation_report usually. 
            # I must update validation_report FIRST.
            context={'temperature_c': environment_params.temperature_c}
        )
        # Update partial result with correct validation info before confidence calculation
        partial_result.validation_report = validation_report
        
        # Re-calculate confidence with validation info
        confidence = self.confidence_scorer.calculate_confidence(
            partial_result,
            context={'temperature_c': environment_params.temperature_c}
        )
        
        # Final Result
        final_result = partial_result
        final_result.confidence_score = confidence
        
        return final_result
