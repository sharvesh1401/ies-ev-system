from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple

# --- Input Parameters ---

class VehicleParameters(BaseModel):
    mass_kg: float = Field(..., gt=0, description="Vehicle mass in kg")
    drag_coefficient: float = Field(..., gt=0, description="Aerodynamic drag coefficient (Cd)")
    frontal_area_m2: float = Field(..., gt=0, description="Frontal area in m^2")
    rolling_resistance_coefficient: float = Field(..., gt=0, description="Rolling resistance coefficient (Cr)")
    battery_capacity_kwh: float = Field(..., gt=0, description="Battery nominal capacity in kWh")
    battery_voltage_nominal: float = Field(..., gt=0, description="Battery nominal voltage in V")
    battery_internal_resistance_ohm: float = Field(..., gt=0, description="Battery internal resistance in Ohm")
    motor_efficiency: float = Field(0.92, ge=0, le=1, description="Motor efficiency")
    regen_efficiency: float = Field(0.70, ge=0, le=1, description="Regenerative braking efficiency")
    auxiliary_power_kw: float = Field(0.5, ge=0, description="Auxiliary power consumption in kW")

class RouteParameters(BaseModel):
    distance_km: float = Field(..., gt=0, description="Total route distance in km")
    elevation_profile: List[Tuple[float, float]] = Field(default=[], description="List of (distance_m, elevation_m) tuples")
    # If elevation_profile is empty, we might use avg_grade
    initial_elevation_m: float = Field(0.0, description="Starting elevation in m")
    target_velocity_mps: float = Field(..., gt=0, description="Target cruising speed in m/s")

class EnvironmentParameters(BaseModel):
    temperature_c: float = Field(25.0, description="Ambient temperature in Celsius")
    wind_speed_mps: float = Field(0.0, ge=0, description="Wind speed in m/s")
    wind_direction_deg: float = Field(0.0, ge=0, le=360, description="Wind direction in degrees (0=North)")
    road_grade_percent: float = Field(0.0, description="Constant road grade % if profile not provided")

# --- Output Results ---

class SimulationStep(BaseModel):
    time_s: float
    distance_m: float
    velocity_mps: float
    acceleration_mps2: float
    elevation_m: float
    grade_percent: float
    
    # Forces
    force_tractive_n: float
    force_aero_n: float
    force_roll_n: float
    force_grade_n: float
    
    # Power & Energy
    power_motor_kw: float
    power_battery_kw: float
    energy_step_kwh: float
    
    # Battery
    soc_percent: float
    voltage_v: float
    current_a: float
    temperature_c: float

class EnergyBreakdown(BaseModel):
    kinetic_kwh: float
    potential_kwh: float
    aerodynamic_kwh: float
    rolling_kwh: float
    auxiliary_kwh: float
    total_kwh: float

class ValidationResult(BaseModel):
    test_name: str
    passed: bool
    details: Dict[str, Any]

class ValidationReport(BaseModel):
    overall_score: float
    tests_passed: int
    total_tests: int
    results: List[ValidationResult]
    interpretation: str

class ConfidenceScore(BaseModel):
    overall: float
    physics_validation: float
    uncertainty: float
    historical_accuracy: float
    interpretation: str
    recommendations: List[str]

class SimulationResult(BaseModel):
    trajectory: List[SimulationStep]
    total_energy_kwh: float
    final_soc_percent: float
    avg_velocity_mps: float
    energy_breakdown: EnergyBreakdown
    validation_report: ValidationReport
    confidence_score: ConfidenceScore
    metadata: Dict[str, Any]
